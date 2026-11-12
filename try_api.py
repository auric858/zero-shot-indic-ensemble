"""Send the full SIB-200 test sets to the running API and compare with the training logs."""
import requests

from config import LABELS
from data_utils import load_test

URL = "http://127.0.0.1:8000/classify"
# Seed-42 test accuracy from the training logs (the served checkpoints are seed-42 models)
EXPECTED = {"santali": 0.662, "bengali": 0.868, "mizo": 0.657, "meitei": 0.377}

for lang, expected in EXPECTED.items():
    texts, y, _ = load_test(lang)
    correct = 0
    for i in range(0, len(texts), 32):
        r = requests.post(URL, json={"texts": texts[i:i + 32], "language": lang}, timeout=300)
        r.raise_for_status()
        body = r.json()
        for pred, true in zip(body["predictions"], y[i:i + 32]):
            correct += pred["label"] == LABELS[true]
    print(f"{lang:8s} model={body['model']:13s} accuracy={correct / len(texts):.3f}"
          f"   (training log: {expected})")
    