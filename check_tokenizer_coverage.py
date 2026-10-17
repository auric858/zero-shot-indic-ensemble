## Measures how heavily each model's tokenizer fragments Santali/Bengali vs English.
import sys
import pandas as pd
from transformers import AutoTokenizer

from config import MODELS, TABLE_DIR, TEST_LANGS, TRAIN_LANG
from data_utils import load_split

sys.stdout.reconfigure(encoding="utf-8")   # let the Windows console print Ol Chiki / Bengali


def tokenizer_stats(tok, texts):
    #returns (tokens_per_word, unk_percent) for a list of sentences.
    n_words = sum(len(t.split()) for t in texts)
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    n_tokens = sum(len(x) for x in ids)
    if tok.unk_token_id is None:
        n_unk = 0
    else:
        n_unk = sum(1 for seq in ids for i in seq if i == tok.unk_token_id)
    return n_tokens / n_words, 100 * n_unk / n_tokens


def main():
    #Same 204 sentences in three languages (English test split is parallel to the others)
    eng_texts, _, _ = load_split(TRAIN_LANG, "test")
    lang_texts = {name: load_split(code, "test")[0] for name, code in TEST_LANGS.items()}
    sample_sat = lang_texts["santali"][0]

    rows = []
    for name, cfg in MODELS.items():
        try:
            tok = AutoTokenizer.from_pretrained(cfg["hf_id"]) ### for each of the 6 models,it downloads the tokenizer only
        except Exception as e:
            print(f"[warn] could not load tokenizer for {name}: {e}")
            continue

        eng_tpw, _ = tokenizer_stats(tok, eng_texts)
        row = {"model": name, "saw_santali": cfg["saw_santali"],
               "eng_tok_per_word": round(eng_tpw, 2)}
        for lang, texts in lang_texts.items():
            tpw, unk = tokenizer_stats(tok, texts)
            row[f"{lang}_tok_per_word"] = round(tpw, 2)
            row[f"{lang}_unk_%"] = round(unk, 2)
            row[f"{lang}_fragmentation_x"] = round(tpw / eng_tpw, 2)
        rows.append(row)

        pieces = tok.tokenize(sample_sat)[:12]
        print(f"{name:13s} first Santali pieces: {pieces}")

    df = pd.DataFrame(rows)
    path = TABLE_DIR / "tokenizer_coverage.csv"
    df.to_csv(path, index=False, encoding="utf-8")

    pd.set_option("display.width", 250, "display.max_columns", 20)
    print("\nTOKENIZER COVERAGE (204 parallel test sentences)")
    print(df.to_string(index=False))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()