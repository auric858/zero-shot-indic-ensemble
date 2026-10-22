"""Hard, soft and weighted soft voting.

Every function takes `probs` with shape (M models, N sentences, C classes)
and returns either predicted class ids (N,) or averaged probabilities (N, C).
"""
import numpy as np


def _check(probs):
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 3:
        raise ValueError("probs must have shape (models, sentences, classes)")
    return probs


def hard_vote(probs):
    """Majority vote of each model's top class. Ties are broken by confidence."""
    probs = _check(probs)
    M, N, C = probs.shape
    preds = probs.argmax(axis=2)                 # (M, N): each model's chosen class
    votes = np.zeros((N, C))                     # how many models chose each class
    conf = np.zeros((N, C))                      # total confidence of the models that chose it
    for m in range(M):
        for n in range(N):
            c = preds[m, n]
            votes[n, c] += 1
            conf[n, c] += probs[m, n, c]
    # conf <= votes <= M, so conf / (M + 1) is always < 1:
    # it can break a tie in votes but can never override a real vote difference.
    return (votes + conf / (M + 1)).argmax(axis=1)


def soft_vote_probs(probs):
    """Average the probability vectors of all models."""
    return _check(probs).mean(axis=0)


def soft_vote(probs):
    return soft_vote_probs(probs).argmax(axis=1)


def normalize_weights(weights, n_models):
    """Validate weights and rescale them to sum to 1."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (n_models,):
        raise ValueError("need exactly one weight per model")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and not all zero")
    return w / w.sum()


def weighted_soft_vote_probs(probs, weights):
    probs = _check(probs)
    w = normalize_weights(weights, probs.shape[0])
    return np.tensordot(w, probs, axes=(0, 0))   # weighted average over the model axis


def weighted_soft_vote(probs, weights):
    return weighted_soft_vote_probs(probs, weights).argmax(axis=1)


if __name__ == "__main__":
    # Quick preview on your real saved predictions (seed 42 only)
    from config import MODELS, PRED_DIR
    from evaluate import metrics

    names = [m for m in MODELS if (PRED_DIR / f"{m}_seed42.npz").exists()]
    data = {m: np.load(PRED_DIR / f"{m}_seed42.npz") for m in names}
    weights = [float(data[m]["val_f1"]) for m in names]     # English validation F1 only
    print("Models:", names)
    print("Weights (English val F1):", [round(w, 3) for w in weights])

    for lang in ("santali", "bengali"):
        probs = np.stack([data[m][f"{lang}_probs"] for m in names])
        y = data[names[0]][f"{lang}_labels"]
        print(f"\n{lang.upper()} (seed 42)")
        rows = [(m, probs[i].argmax(1)) for i, m in enumerate(names)]
        rows += [("HARD VOTE", hard_vote(probs)),
                 ("SOFT VOTE", soft_vote(probs)),
                 ("WEIGHTED SOFT VOTE", weighted_soft_vote(probs, weights))]
        for label, pred in rows:
            r = metrics(y, pred)
            print(f"  {label:20s} acc {r['accuracy']:.3f}   macro-F1 {r['macro_f1']:.3f}")