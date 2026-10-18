"""Inference (softmax probabilities) and metrics."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


@torch.no_grad()
def predict_probs(model, tok, texts, device, max_len=128, batch_size=32):
    """Return an array of shape (num_sentences, num_classes) of softmax probabilities."""
    model.eval()
    out = []
    use_amp = device.type == "cuda"
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(**enc).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def metrics(y_true, y_pred):
    """Accuracy and macro-F1 (the average of per-class F1, so small classes count equally)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


if __name__ == "__main__":
    y = [0, 1, 2, 2]
    print(metrics(y, [0, 1, 2, 2]))   # perfect -> 1.0 / 1.0
    print(metrics(y, [0, 0, 0, 0]))   # always class 0 -> low macro-F1