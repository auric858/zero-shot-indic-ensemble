"""EXPLORATORY translate-then-classify pipeline (never mixed with the main tables).

  1) NLLB-200 translates the source text into English
     (or you supply ready-made English with --english-col).
  2) An English-only fine-tuned classifier (outputs/models/<name>) predicts the topic.

Modes
  python mundari_extension.py --demo-santali
      Validates the pipeline on SIB-200 Santali (NLLB supports sat_Olck), where
      real labels exist. Compare with the direct zero-shot numbers.
  python mundari_extension.py --csv data/my.csv --text-col mundari --src hin_Deva
      Your own text, translated with NLLB. If the CSV has a 'label' column,
      metrics are reported; otherwise only translations and predictions are saved.
  python mundari_extension.py --csv data/my.csv --text-col mundari --english-col english
      Skips NLLB and classifies the English you supplied (an "English control").

NLLB-200 has no Mundari language code. --src must be a PROXY code (e.g. hin_Deva
for Devanagari text, ben_Beng for Bengali script, ory_Orya for Odia script, or
sat_Olck). Expect poor translations: the result measures the translator as much
as the classifier.
"""
import argparse
import sys

import pandas as pd
import torch
from transformers import (AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
                          AutoTokenizer)

from check_tokenizer_coverage import tokenizer_stats
from config import NLLB_ID, MODEL_DIR, TABLE_DIR, LABELS, LABEL2ID, MODELS
from data_utils import load_test
from evaluate import predict_probs, metrics
from models import get_device

sys.stdout.reconfigure(encoding="utf-8")


def coverage_report(texts):
    """Which of the six tokenizers can even read this text? (label-free)"""
    rows = []
    for name, cfg in MODELS.items():
        tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
        tpw, unk = tokenizer_stats(tok, texts)
        rows.append({"model": name, "tokens_per_word": round(tpw, 2), "unk_%": round(unk, 1)})
    print("\nTokenizer coverage on this text:")
    print(pd.DataFrame(rows).to_string(index=False))


def translate(texts, src_lang, device, batch_size=16):
    """Translate to English with NLLB-200 (600M distilled)."""
    tok = AutoTokenizer.from_pretrained(NLLB_ID, src_lang=src_lang)
    mt = AutoModelForSeq2SeqLM.from_pretrained(NLLB_ID).to(device).eval()
    english_id = tok.convert_tokens_to_ids("eng_Latn")
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to(device)
        with torch.no_grad():
            gen = mt.generate(**enc, forced_bos_token_id=english_id,
                              max_new_tokens=128, num_beams=4)
        out += tok.batch_decode(gen, skip_special_tokens=True)
        print(f"  translated {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    del mt
    torch.cuda.empty_cache()
    return out


def classify(english_texts, clf_name, device):
    path = MODEL_DIR / clf_name
    if not path.exists():
        raise SystemExit(f"No saved checkpoint at {path}. Tell me and we will add a "
                         f"--force flag to train.py to re-save it.")
    tok = AutoTokenizer.from_pretrained(path)
    clf = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    probs = predict_probs(clf, tok, english_texts, device)
    del clf
    torch.cuda.empty_cache()
    return probs.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-santali", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--english-col", help="column of ready-made English (skips NLLB)")
    ap.add_argument("--src", default="hin_Deva", help="NLLB proxy source-language code")
    ap.add_argument("--clf", default="xlmr_base",
                    help="English-trained classifier in outputs/models")
    a = ap.parse_args()
    device = get_device()

    y = None
    english_given = None
    if a.demo_santali:
        texts, y, _ = load_test("santali")
        src, tag = "sat_Olck", "santali_translate_classify"
    elif a.csv:
        df = pd.read_csv(a.csv, encoding="utf-8-sig")
        df = df.dropna(subset=[a.text_col])                          # drop empty cells
        df = df[df[a.text_col].astype(str).str.strip() != ""]        # drop blank strings
        texts = df[a.text_col].astype(str).tolist()
        english_given = df[a.english_col].astype(str).tolist() if a.english_col else None
        if "label" in df:
            y = [LABEL2ID[l] for l in df["label"]]
        src = a.src
        tag = ("mundari_english_control" if english_given is not None
               else "mundari_translate_classify")
        coverage_report(texts)
    else:
        raise SystemExit("Use --demo-santali or --csv FILE")

    print(f"\n{len(texts)} sentences | source code: {src} | classifier: {a.clf}")
    english = english_given if english_given is not None else translate(texts, src, device)
    pred = classify(english, a.clf, device)

    out = pd.DataFrame({"english_translation": english,
                        "predicted": [LABELS[p] for p in pred]})
    if y is not None:
        out["true"] = [LABELS[i] for i in y]
    path = TABLE_DIR / f"{tag}_predictions.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved {path}")
    print("\nFirst 5 rows:")
    print(out.head().to_string())

    if y is not None:
        m = metrics(y, pred)
        print(f"\n[{tag}] accuracy {m['accuracy']:.3f}   macro-F1 {m['macro_f1']:.3f}")
        if a.demo_santali:
            ref = pd.read_csv(TABLE_DIR / "main_results_santali.csv", index_col=0)
            print("\nFor comparison, DIRECT zero-shot on Santali (mean of 3 seeds):")
            print(ref.loc[["indicbert_v2", "glot500", "majority_class"],
                          ["acc_mean", "f1_mean"]])
    else:
        print("\nNo labels: nothing to score. Read the translations to judge quality.")


if __name__ == "__main__":
    main()