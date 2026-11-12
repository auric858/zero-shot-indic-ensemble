"""Detect the dominant writing system of a text from Unicode character names.

No model is needed. This is the one reliable signal for Santali: Ol Chiki is
written in its own script. Bengali script is shared by Bengali, Assamese and
Meitei (as used in SIB-200), so it cannot tell those languages apart.
"""
import unicodedata
from collections import Counter

# Unicode character names start with their script, e.g. "OL CHIKI LETTER LA"
SCRIPT_PREFIXES = {
    "OL CHIKI": "ol_chiki",
    "BENGALI": "bengali",
    "DEVANAGARI": "devanagari",
    "MEETEI MAYEK": "meitei_mayek",
    "LATIN": "latin",
}


def detect_script(text: str):
    """Return (script, share) where share is the fraction of letters in that script.

    Digits, spaces and punctuation are ignored. Scripts we do not name count as
    'other'. A text with no letters returns ('unknown', 0.0).
    """
    counts = Counter()
    for ch in text:
        if not (ch.isalpha() or unicodedata.category(ch).startswith("M")):
            continue
        name = unicodedata.name(ch, "")
        for prefix, script in SCRIPT_PREFIXES.items():
            if name.startswith(prefix):
                counts[script] += 1
                break
        else:
            counts["other"] += 1
    if not counts:
        return "unknown", 0.0
    script, n = counts.most_common(1)[0]
    return script, n / sum(counts.values())
