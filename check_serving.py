"""Why does the API differ from the training log by one sentence on Bengali and Mizo?"""
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MODEL_DIR, PRED_DIR
from data_utils import load_test

CASES = [("xlmr_large", "bengali"), ("glot500", "mizo")]


def predict(model, tok, texts, batch_size, bf16):
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                      max_length=128, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                logits = model(**enc).logits
            out.append(logits.float().argmax(-1).cpu().numpy())
    return np.concatenate(out)


for name, lang in CASES:
    texts, y, _ = load_test(lang)
    y = np.array(y)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR / name)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / name).to("cuda").eval()
    orig = np.load(PRED_DIR / f"{name}_seed42.npz")[f"{lang}_probs"].argmax(1)
    print(f"\n{name} on {lang}: original training-log accuracy {np.mean(orig == y):.3f}")
    for bf16 in (True, False):
        for bs in (32, 16, 1):
            p = predict(model, tok, texts, bs, bf16)
            print(f"  bf16={str(bf16):5s} batch={bs:2d}  accuracy {np.mean(p == y):.3f}"
                  f"   differs from original on {int(np.sum(p != orig))} sentences")
    del model
    torch.cuda.empty_cache()