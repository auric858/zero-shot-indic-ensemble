# Loading the English (train/val) and Santali/Bengali (test) splits of SIB-200.
from functools import lru_cache

from datasets import load_dataset

from config import DATASET_ID, LABELS, LABEL2ID, TRAIN_LANG, TEST_LANGS


@lru_cache(maxsize=None)
def load_split(lang_code: str, split: str):
    """Return (texts, label_ids, index_ids) for one language and one split.

    lang_code : SIB-200 config name, e.g. 'eng_Latn', 'sat_Olck', 'ben_Beng'
    split     : 'train', 'validation' or 'test'
    """
    ds = load_dataset(DATASET_ID, lang_code, split=split, trust_remote_code=True)
    ## checking for unknown categories which are not in our set labels..
    unknown = set(ds["category"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected labels in {lang_code}/{split}: {unknown}")

    texts = list(ds["text"])
    labels = [LABEL2ID[c] for c in ds["category"]]
    index_ids = list(ds["index_id"])
    return texts, labels, index_ids


def load_english():
    # English train and validation. These are the only labeled data the models see.
    train = load_split(TRAIN_LANG, "train")
    val = load_split(TRAIN_LANG, "validation")
    return train, val


def load_test(lang_name: str):
    # Zero-shot test set for 'santali' or 'bengali'.
    return load_split(TEST_LANGS[lang_name], "test")


if __name__ == "__main__":
    from collections import Counter

    (tr_x, tr_y, _), (va_x, va_y, _) = load_english()
    print(f"English train      : {len(tr_x)} sentences")
    print(f"English validation : {len(va_x)} sentences")

    tests = {name: load_test(name) for name in TEST_LANGS}
    for name, (x, y, _) in tests.items():
        print(f"{name.capitalize():8s} test      : {len(x)} sentences")

    # Same sentences, different languages?
    ids = [tuple(t[2]) for t in tests.values()]
    print("Test sets are parallel (same sentence IDs):", all(i == ids[0] for i in ids))
    print("Labels identical across test languages    :",
          all(t[1] == list(tests.values())[0][1] for t in tests.values()))

    print("\nLabel distribution in the test set:")
    for label_id, n in sorted(Counter(list(tests.values())[0][1]).items()):
        print(f"  {LABELS[label_id]:20s} {n}")

    print("\nExample sentence in each language:")
    print("  English :", tr_x[0][:90])
    for name, (x, y, _) in tests.items():
        print(f"  {name.capitalize():8s}:", x[0][:90], "->", LABELS[y[0]])