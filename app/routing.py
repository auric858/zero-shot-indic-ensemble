"""Which model serves which language, and what we honestly know about it.

Routes follow the SIB-200 benchmark results (mean over 3 seeds, 204 test
sentences). The served checkpoint is the seed-42 model.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Route:
    model: str
    scripts: frozenset
    reliability: str          # "good" | "moderate" | "low"
    benchmark: str
    note: str = ""


ROUTES = {
    "santali": Route("indicbert_v2", frozenset({"ol_chiki"}), "moderate",
                     "SIB-200 test macro-F1 0.625 (majority baseline 0.057)"),
    "mizo": Route("glot500", frozenset({"latin"}), "moderate",
                  "SIB-200 test macro-F1 0.650 (majority baseline 0.057)"),
    "bengali": Route("xlmr_large", frozenset({"bengali"}), "good",
                     "SIB-200 test macro-F1 0.863 (majority baseline 0.057)"),
    "english": Route("xlmr_large", frozenset({"latin"}), "good",
                     "English validation macro-F1 0.876 (99 sentences)"),
    "meitei": Route("indicbert_v2", frozenset({"bengali"}), "low",
                    "SIB-200 test: best accuracy 0.337, macro-F1 about 0.25 to 0.27 "
                    "(majority baseline accuracy 0.25)",
                    "Meitei results were close to chance for every model tested. "
                    "Treat this prediction as unreliable."),
}

# Only these scripts are safe to map to a language without being told.
AUTO_SCRIPT = {"ol_chiki": "santali", "bengali": "bengali", "latin": "english"}
AUTO_NOTES = {
    "bengali": "Bengali-script text was assumed to be Bengali. Meitei and Assamese "
               "also use Bengali script; pass language='meitei' if that is what you sent.",
    "english": "Latin-script text was assumed to be English. "
               "Pass language='mizo' for Mizo.",
}


class RoutingError(ValueError):
    """The request cannot be routed to an evaluated model."""


@dataclass
class Decision:
    language: str
    route: Route
    script: str
    warnings: list = field(default_factory=list)


def resolve(requested: str, detections):
    """detections: list of (script, share), one per input text."""
    requested = (requested or "auto").lower()
    warnings = []
    scripts = sorted({s for s, _ in detections})

    if requested == "auto":
        langs = set()
        for script, _ in detections:
            if script not in AUTO_SCRIPT:
                raise RoutingError(
                    f"Cannot detect a supported language from script '{script}'. "
                    f"Auto-detection covers Ol Chiki (Santali), Bengali script and "
                    f"Latin script. Otherwise pass language explicitly: "
                    f"{', '.join(sorted(ROUTES))}.")
            langs.add(AUTO_SCRIPT[script])
        if len(langs) > 1:
            raise RoutingError("The texts appear to be in different languages. "
                               "Send one language per request.")
        language = langs.pop()
        if language in AUTO_NOTES:
            warnings.append(AUTO_NOTES[language])
    else:
        if requested not in ROUTES:
            raise RoutingError(f"Unknown language '{requested}'. "
                               f"Choose from: {', '.join(sorted(ROUTES))} or 'auto'.")
        language = requested
        expected = ROUTES[language].scripts
        if any(s not in expected for s in scripts):
            warnings.append(f"Detected script {scripts} does not match the usual "
                            f"script for '{language}' ({sorted(expected)}). "
                            f"The prediction may be meaningless.")

    if min(share for _, share in detections) < 0.6:
        warnings.append("The text mixes writing systems or contains unrecognised "
                        "characters, so the model may be reading little of it.")
    route = ROUTES[language]
    if route.note:
        warnings.append(route.note)
    script = scripts[0] if len(scripts) == 1 else "mixed"
    return Decision(language, route, script, warnings)