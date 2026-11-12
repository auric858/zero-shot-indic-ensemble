"""FastAPI demo: zero-shot topic classification for low-resource Indian languages."""
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.classifier import ModelNotAvailable, ModelStore
from app.routing import ROUTES, RoutingError, resolve
from app.script_detect import detect_script

app = FastAPI(
    title="Zero-shot topic classification for low-resource Indian languages",
    description=("Models are fine-tuned on English only and evaluated zero-shot on the "
                 "SIB-200 benchmark. Every response says which model answered and how "
                 "reliable it was on the benchmark."),
    version="1.0.0",
)

_store = None


def get_store():
    global _store
    if _store is None:
        _store = ModelStore()
    return _store


class ClassifyRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32,
                             description="1 to 32 sentences, all in the same language")
    language: str = Field("auto", description="auto, santali, mizo, bengali, english or meitei")

    @field_validator("texts")
    @classmethod
    def check_texts(cls, texts):
        for t in texts:
            if not t.strip():
                raise ValueError("texts must not contain empty strings")
            if len(t) > 2000:
                raise ValueError("each text must be at most 2000 characters")
        return texts


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class ClassifyResponse(BaseModel):
    language: str
    script: str
    model: str
    reliability: str
    benchmark: str
    warnings: list[str]
    predictions: list[Prediction]


@app.get("/health")
def health(store=Depends(get_store)):
    return {"status": "ok", "models_available": store.available()}


@app.get("/languages")
def languages(store=Depends(get_store)):
    have = set(store.available())
    return {name: {"model": r.model, "model_available": r.model in have,
                   "reliability": r.reliability, "benchmark": r.benchmark}
            for name, r in ROUTES.items()}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest, store=Depends(get_store)):
    detections = [detect_script(t) for t in req.texts]
    try:
        decision = resolve(req.language, detections)
    except RoutingError as e:
        raise HTTPException(status_code=422, detail=str(e))

    route = decision.route
    try:
        labels, probs = store.predict(route.model, req.texts)
    except ModelNotAvailable:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{route.model}' is not installed on this server.")

    predictions = []
    for row in probs:
        k = int(row.argmax())
        predictions.append(Prediction(
            label=labels[k], confidence=round(float(row[k]), 4),
            probabilities={l: round(float(p), 4) for l, p in zip(labels, row)}))
    return ClassifyResponse(language=decision.language, script=decision.script,
                            model=route.model, reliability=route.reliability,
                            benchmark=route.benchmark, warnings=decision.warnings,
                            predictions=predictions)
