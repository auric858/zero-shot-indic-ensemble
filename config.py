"""Central configuration: paths, languages, labels, model registry, training defaults."""
from pathlib import Path

# paths
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PRED_DIR = OUT / "predictions"
MODEL_DIR = OUT / "models"
TABLE_DIR = OUT / "tables"
for _d in (PRED_DIR, MODEL_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# data
DATASET_ID = "Davlan/sib200"     # SIB-200 dataset on the Hugging Face Hub
TRAIN_LANG = "eng_Latn"          # train + validation will use English only

# Zero-shot test languages: short name = SIB-200 language code
## other langs can be added later
TEST_LANGS = {
    "santali": "sat_Olck",       # Ol Chiki script, unseen by most models (hard case)
    "bengali": "ben_Beng",       # seen by all six models (easy control case)
}

LABELS = ["science/technology", "travel", "politics", "sports",
          "health", "entertainment", "geography"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# models
MODELS = {
    "mbert":        {"hf_id": "bert-base-multilingual-cased",
                     "saw_santali": "No", "saw_bengali": "Yes"},
    "xlmr_base":    {"hf_id": "xlm-roberta-base",
                     "saw_santali": "No", "saw_bengali": "Yes"},
    "xlmr_large":   {"hf_id": "xlm-roberta-large",
                     "saw_santali": "No", "saw_bengali": "Yes",
                     "overrides": {"batch_size": 4, "grad_accum": 4, "lr": 1e-5,
                                   "freeze_embeddings": True}},
    "muril":        {"hf_id": "google/muril-base-cased",
                     "saw_santali": "No", "saw_bengali": "Yes"},
    "glot500":      {"hf_id": "cis-lmu/glot500-base",
                     "saw_santali": "Uncertain", "saw_bengali": "Yes"},
    "indicbert_v2": {"hf_id": "ai4bharat/IndicBERTv2-MLM-Sam-TLM",
                     "saw_santali": "Yes", "saw_bengali": "Yes"},
}

# training defaults
DEFAULTS = dict(
    max_len=128,        # max tokens per sentence
    epochs=8,
    batch_size=16,
    grad_accum=1,       # >1 simulates a bigger batch on a small GPU
    lr=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
)
SEEDS = [42, 43, 44]    # repeat every run 3 times, report mean and std

# Mundari
NLLB_ID = "facebook/nllb-200-distilled-600M"   # Meta's translation model for the Mundari extension


if __name__ == "__main__":
    print("Project root :", ROOT)
    print("Labels       :", len(LABELS))
    print("Test langs   :", TEST_LANGS)
    for name, cfg in MODELS.items():
        print(f"  {name:13s} -> {cfg['hf_id']}")