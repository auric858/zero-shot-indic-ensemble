"""Compare individual models and ensembles on Santali and Bengali.

Produces, per language (CSV files in outputs/tables/):
  main_results_<lang>.csv   accuracy / macro-F1, mean and std over seeds
  per_class_f1_<lang>.csv   per-class F1 (mean over seeds)
  mcnemar_<lang>.csv        best single model vs best ensemble, per seed
  ablations_<lang>.csv      which models go into the ensemble
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

from config import MODELS, PRED_DIR, TABLE_DIR, TEST_LANGS, LABELS
from data_utils import load_english
import ensemble as E
from evaluate import metrics

ENSEMBLES = ["hard_vote", "soft_vote", "weighted_soft_vote"]
SEEDS_TO_USE = [42, 43, 44]

pd.set_option("display.width", 200, "display.max_columns", 30)


# ------------------------------------------------------------------ loading
def load_seed(seed):
    """Load every model's saved predictions for one seed into plain dicts."""
    data = {}
    for m in MODELS:
        f = PRED_DIR / f"{m}_seed{seed}.npz"
        if f.exists():
            with np.load(f) as z:
                data[m] = {k: z[k] for k in z.files}
    return data


def readable_models(threshold=10.0):
    """Models whose tokenizer can read Ol Chiki (Santali UNK% below threshold).

    The rule uses only tokenizer statistics computed in Step 4. It uses no
    Santali labels, so it does not break the zero-shot setup.
    """
    path = TABLE_DIR / "tokenizer_coverage.csv"
    if not path.exists():
        raise SystemExit("Run check_tokenizer_coverage.py first (Step 4).")
    df = pd.read_csv(path)
    return df.loc[df["santali_unk_%"] < threshold, "model"].tolist()


# ------------------------------------------------------------- predictions
def ensemble_preds(data, lang, names):
    """The three ensembles built from a chosen subset of models."""
    probs = np.stack([data[m][f"{lang}_probs"] for m in names])
    weights = [float(data[m]["val_f1"]) for m in names]   # English validation F1 only
    return {
        "hard_vote": E.hard_vote(probs),
        "soft_vote": E.soft_vote(probs),
        "weighted_soft_vote": E.weighted_soft_vote(probs, weights),
    }


def all_system_preds(data, lang, majority_class, n_sentences):
    """Six individual models + three ensembles + majority-class baseline."""
    names = list(data)
    preds = {m: data[m][f"{lang}_probs"].argmax(1) for m in names}
    preds.update(ensemble_preds(data, lang, names))
    preds["majority_class"] = np.full(n_sentences, majority_class)
    return preds


# ------------------------------------------------------------ one language
def run_language(lang, seed_data, majority_class):
    seeds = list(seed_data)
    rows, pc_rows, preds_by_seed = [], [], {}
    y = None
    for s in seeds:
        data = seed_data[s]
        y = data[next(iter(data))][f"{lang}_labels"]
        preds = all_system_preds(data, lang, majority_class, len(y))
        preds_by_seed[s] = preds
        for name, p in preds.items():
            rows.append({"system": name, "seed": s, **metrics(y, p)})
            f1s = f1_score(y, p, average=None, labels=list(range(len(LABELS))),
                           zero_division=0)
            pc_rows.append({"system": name, "seed": s, **dict(zip(LABELS, f1s))})

    # ---- main table: mean and std over seeds
    df = pd.DataFrame(rows)
    agg = df.groupby("system")[["accuracy", "macro_f1"]].agg(["mean", "std"])
    agg.columns = ["acc_mean", "acc_std", "f1_mean", "f1_std"]
    kind = {m: "single" for m in MODELS}
    kind.update({e: "ensemble" for e in ENSEMBLES})
    kind["majority_class"] = "baseline"
    agg.insert(0, "type", [kind[s] for s in agg.index])
    agg = agg.sort_values("f1_mean", ascending=False).round(3)
    agg.to_csv(TABLE_DIR / f"main_results_{lang}.csv")
    print(f"\n{'=' * 70}\n  {lang.upper()}: mean +- std over {len(seeds)} seeds "
          f"({len(y)} test sentences)\n{'=' * 70}")
    print(agg)

    # ---- per-class F1
    pc = pd.DataFrame(pc_rows).groupby("system")[LABELS].mean().round(3)
    pc = pc.loc[agg.index]
    pc.to_csv(TABLE_DIR / f"per_class_f1_{lang}.csv")

    best_single = agg[agg["type"] == "single"].index[0]
    best_ens = agg[agg["type"] == "ensemble"].index[0]
    print(f"\nBest single model : {best_single}   |   best ensemble : {best_ens}")
    print("\nPer-class F1 (mean over seeds):")
    print(pc.loc[[best_single, best_ens]])

    # ---- McNemar: best single vs best ensemble, one test per seed
    mc_rows = []
    for s in seeds:
        a = preds_by_seed[s][best_single] == y     # single model correct?
        b = preds_by_seed[s][best_ens] == y        # ensemble correct?
        only_single = int(np.sum(a & ~b))
        only_ens = int(np.sum(~a & b))
        table = [[int(np.sum(a & b)), only_single],
                 [only_ens, int(np.sum(~a & ~b))]]
        p = mcnemar(table, exact=True).pvalue
        mc_rows.append({"seed": s, "best_single": best_single, "best_ensemble": best_ens,
                        "only_single_correct": only_single,
                        "only_ensemble_correct": only_ens, "p_value": round(p, 4)})
    mc = pd.DataFrame(mc_rows)
    mc.to_csv(TABLE_DIR / f"mcnemar_{lang}.csv", index=False)
    print("\nMcNemar exact test (best single vs best ensemble):")
    print(mc.to_string(index=False))

    # ---- ablations: which models go into the ensemble?
    readable = readable_models()
    configs = {"all_six": list(MODELS)}
    configs["readable_only: " + "+".join(readable)] = readable
    for m in MODELS:
        configs[f"leave_out_{m}"] = [x for x in MODELS if x != m]
    ab_rows = []
    for cname, subset in configs.items():
        for s in seeds:
            for ename, p in ensemble_preds(seed_data[s], lang, subset).items():
                ab_rows.append({"config": cname, "ensemble": ename, "seed": s,
                                **metrics(y, p)})
    ab = (pd.DataFrame(ab_rows).groupby(["config", "ensemble"])[["accuracy", "macro_f1"]]
          .mean().round(3).reset_index())
    ab.to_csv(TABLE_DIR / f"ablations_{lang}.csv", index=False)
    print("\nAblations (mean over seeds):")
    print(ab.pivot(index="config", columns="ensemble", values="macro_f1")
            [ENSEMBLES].round(3))


# --------------------------------------------------------------------- main
def main():
    seed_data = {}
    for s in SEEDS_TO_USE:
        d = load_seed(s)
        if len(d) == len(MODELS):
            seed_data[s] = d
        else:
            print(f"[warn] seed {s}: only {len(d)}/{len(MODELS)} models found, skipped")
    if not seed_data:
        raise SystemExit("No complete seeds found. Run train.py first.")

    # Majority class comes from ENGLISH TRAIN labels, never from the test set
    (_, train_y, _), _ = load_english()
    majority_class = Counter(train_y).most_common(1)[0][0]
    print(f"Majority-class baseline predicts: {LABELS[majority_class]}")

    for lang in TEST_LANGS:
        run_language(lang, seed_data, majority_class)
    print(f"\nTables saved to {TABLE_DIR}")


if __name__ == "__main__":
    main()
    