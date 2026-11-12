import pathlib
import sys

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.main import app, get_store
from app.script_detect import detect_script

LABELS = ["science/technology", "travel", "politics", "sports",
          "health", "entertainment", "geography"]

SANTALI = "ᱢᱤᱭᱩᱴᱮᱥᱚᱱ ᱱᱟᱶᱟ ᱡᱮᱱᱮᱴᱤᱠ"
BENGALI = "মিউটেশন নতুন জিনগত প্রকরণ যুক্ত করে"
ENGLISH = "The nucleus is made up of two particles."


class FakeStore:
    """Stands in for the real models: always answers 'sports' with 0.9."""
    def __init__(self, have=("indicbert_v2", "glot500", "xlmr_large")):
        self.have = have
        self.calls = []

    def available(self):
        return list(self.have)

    def predict(self, name, texts):
        from app.classifier import ModelNotAvailable
        if name not in self.have:
            raise ModelNotAvailable(name)
        self.calls.append(name)
        probs = np.full((len(texts), 7), 0.1 / 6)
        probs[:, 3] = 0.9
        return LABELS, probs


def make_client(store=None):
    store = store or FakeStore()
    app.dependency_overrides[get_store] = lambda: store
    return TestClient(app), store


def test_script_detection():
    assert detect_script(SANTALI)[0] == "ol_chiki"
    assert detect_script(BENGALI)[0] == "bengali"
    assert detect_script(ENGLISH)[0] == "latin"
    assert detect_script("12345 !!!") == ("unknown", 0.0)


def test_auto_routes_each_script_to_its_model():
    c, store = make_client()
    for text, model, lang in [(SANTALI, "indicbert_v2", "santali"),
                              (BENGALI, "xlmr_large", "bengali"),
                              (ENGLISH, "xlmr_large", "english")]:
        r = c.post("/classify", json={"texts": [text]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == model and body["language"] == lang
        assert body["predictions"][0]["label"] == "sports"
        assert abs(sum(body["predictions"][0]["probabilities"].values()) - 1) < 1e-3


def test_explicit_language_overrides_auto():
    c, _ = make_client()
    body = c.post("/classify", json={"texts": [ENGLISH], "language": "mizo"}).json()
    assert body["model"] == "glot500" and body["language"] == "mizo"


def test_meitei_is_flagged_unreliable():
    c, _ = make_client()
    body = c.post("/classify", json={"texts": [BENGALI], "language": "meitei"}).json()
    assert body["reliability"] == "low"
    assert any("chance" in w for w in body["warnings"])


def test_script_mismatch_warns():
    c, _ = make_client()
    body = c.post("/classify", json={"texts": [ENGLISH], "language": "santali"}).json()
    assert any("does not match" in w for w in body["warnings"])


def test_auto_rejects_unsupported_script_and_mixed_languages():
    c, _ = make_client()
    assert c.post("/classify", json={"texts": ["यह हिंदी है"]}).status_code == 422
    assert c.post("/classify", json={"texts": [SANTALI, ENGLISH]}).status_code == 422


def test_validation_errors():
    c, _ = make_client()
    assert c.post("/classify", json={"texts": []}).status_code == 422
    assert c.post("/classify", json={"texts": ["  "]}).status_code == 422
    assert c.post("/classify", json={"texts": ["x" * 2001]}).status_code == 422
    assert c.post("/classify", json={"texts": [ENGLISH], "language": "klingon"}).status_code == 422


def test_missing_model_gives_503():
    c, _ = make_client(FakeStore(have=("glot500",)))
    assert c.post("/classify", json={"texts": [SANTALI]}).status_code == 503


def test_health_and_languages():
    c, _ = make_client(FakeStore(have=("glot500",)))
    assert c.get("/health").json()["models_available"] == ["glot500"]
    langs = c.get("/languages").json()
    assert langs["mizo"]["model_available"] is True
    assert langs["santali"]["model_available"] is False\
    