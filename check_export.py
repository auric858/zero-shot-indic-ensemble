"""Did the export retraining reproduce the original seed-42 runs?"""
import numpy as np

from config import PRED_DIR

for m in ("indicbert_v2", "glot500", "xlmr_large"):
    old = np.load(PRED_DIR / f"{m}_seed42.npz")
    new = np.load(PRED_DIR / f"export_{m}_seed42.npz")
    for lang in ("santali", "bengali", "mizo", "meitei"):
        y = old[f"{lang}_labels"]
        p_old = old[f"{lang}_probs"].argmax(1)
        p_new = new[f"{lang}_probs"].argmax(1)
        print(f"{m:13s} {lang:8s} original {np.mean(p_old == y):.3f}   "
              f"export {np.mean(p_new == y):.3f}   "
              f"identical predictions on {np.mean(p_old == p_new):.1%} of sentences")
        