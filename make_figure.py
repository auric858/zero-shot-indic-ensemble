"""Heatmap of macro-F1 for every system on every test language (README figure)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import OUT, TABLE_DIR, TEST_LANGS

FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ROWS = [("mbert", "mBERT"), ("xlmr_base", "XLM-R base"), ("xlmr_large", "XLM-R large"),
        ("muril", "MuRIL"), ("glot500", "Glot500"), ("indicbert_v2", "IndicBERT v2"),
        ("hard_vote", "Hard vote (all 6)"), ("soft_vote", "Soft vote (all 6)"),
        ("majority_class", "Majority baseline")]
COLS = {"santali": "Santali\n(Ol Chiki)", "bengali": "Bengali", "mizo": "Mizo\n(Latin)",
        "meitei": "Meitei\n(Bengali script)"}


def main():
    f1 = {}
    for lang in TEST_LANGS:
        df = pd.read_csv(TABLE_DIR / f"main_results_{lang}.csv", index_col=0)
        f1[lang] = df["f1_mean"]
    table = pd.DataFrame(f1).loc[[r for r, _ in ROWS], list(COLS)]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow(table.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(COLS)), labels=list(COLS.values()))
    ax.set_yticks(range(len(ROWS)), labels=[name for _, name in ROWS])
    ax.xaxis.tick_top()
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v < 0.55 else "black", fontsize=9)
    # mark the best single model per language
    singles = [r for r, _ in ROWS[:6]]
    for j, lang in enumerate(COLS):
        best = table.loc[singles, lang].idxmax()
        i = [r for r, _ in ROWS].index(best)
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="red", lw=2.5))
    ax.set_title("Zero-shot macro-F1 (English-only training, mean of 3 seeds)\n"
                 "red box = best single model", pad=40, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="macro-F1")
    fig.tight_layout()
    path = FIG_DIR / "f1_heatmap.png"
    fig.savefig(path, dpi=160)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()