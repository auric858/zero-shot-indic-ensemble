"""Paired McNemar test: translate-then-classify (Santali) vs direct zero-shot models.

Needs outputs/tables/santali_translate_classify_predictions.csv from
`python mundari_extension.py --demo-santali`.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from config import TABLE_DIR, PRED_DIR, LABELS

L2I = {label: i for i, label in enumerate(LABELS)}


def main():
    df = pd.read_csv(TABLE_DIR / "santali_translate_classify_predictions.csv", encoding="utf-8")
    y = np.array([L2I[t] for t in df["true"]])
    translate_pred = np.array([L2I[p] for p in df["predicted"]])
    print(f"Translate-then-classify accuracy: {(translate_pred == y).mean():.3f}")

    rows = []
    for model in ["indicbert_v2", "glot500"]:
        for seed in (42, 43, 44):
            with np.load(PRED_DIR / f"{model}_seed{seed}.npz") as z:
                assert (z["santali_labels"] == y).all(), "sentence order mismatch"
                direct_pred = z["santali_probs"].argmax(1)
            a = translate_pred == y            # translation route correct?
            b = direct_pred == y               # direct model correct?
            only_t = int(np.sum(a & ~b))
            only_d = int(np.sum(~a & b))
            table = [[int(np.sum(a & b)), only_t], [only_d, int(np.sum(~a & ~b))]]
            rows.append({"direct_model": model, "seed": seed,
                         "direct_accuracy": round(b.mean(), 3),
                         "only_translate_correct": only_t,
                         "only_direct_correct": only_d,
                         "p_value": round(mcnemar(table, exact=True).pvalue, 4)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "mcnemar_translate_vs_direct_santali.csv", index=False)
    print("\nMcNemar exact test (translate-then-classify vs direct zero-shot):")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()