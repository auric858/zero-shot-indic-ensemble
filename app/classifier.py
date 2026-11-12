"""Lazy model loading and inference for the API.

Models are loaded the first time they are needed and kept in a small LRU cache,
so a laptop or a small container never holds all three in memory at once.
torch and transformers are imported inside functions so that the rest of the
app (and its tests) work without them.
"""
import gc
import os
import threading
from collections import OrderedDict
from pathlib import Path


class ModelNotAvailable(RuntimeError):
    pass


class ModelStore:
    def __init__(self, model_dir=None, max_loaded=None, max_len=128, batch_size=16):
        self.model_dir = Path(model_dir or os.getenv("MODEL_DIR", "outputs/models"))
        self.max_loaded = int(max_loaded or os.getenv("MAX_LOADED_MODELS", 2))
        self.max_len = max_len
        self.batch_size = batch_size
        self._loaded = OrderedDict()
        self._lock = threading.Lock()

    def available(self):
        if not self.model_dir.exists():
            return []
        return sorted(p.name for p in self.model_dir.iterdir()
                      if (p / "config.json").exists())

    def _get(self, name):
        with self._lock:
            if name in self._loaded:
                self._loaded.move_to_end(name)
                return self._loaded[name]
            path = self.model_dir / name
            if not (path / "config.json").exists():
                raise ModelNotAvailable(name)
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model = model.to(device).eval()
            self._loaded[name] = (tok, model)
            while len(self._loaded) > self.max_loaded:
                self._loaded.popitem(last=False)
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
            return self._loaded[name]

    def predict(self, name, texts):
        """Return (label_names, probabilities with shape (len(texts), n_labels))."""
        import numpy as np
        import torch
        tok, model = self._get(name)
        device = next(model.parameters()).device
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                          max_length=self.max_len, return_tensors="pt").to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    logits = model(**enc).logits
                out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        probs = np.concatenate(out)
        id2label = model.config.id2label
        return [id2label[i] for i in range(probs.shape[1])], probs
    