## Builds a tokenizer + classification model from a name in config.MODELS.
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MODELS, LABELS, ID2LABEL, LABEL2ID


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(name: str):
    """Return (tokenizer, model) for one of the six models."""
    hf_id = MODELS[name]["hf_id"]
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return tok, model


if __name__ == "__main__":
    tok, model = build("mbert")
    n = sum(p.numel() for p in model.parameters())
    print(f"mbert loaded: {n/1e6:.0f}M parameters, device = {get_device()}")