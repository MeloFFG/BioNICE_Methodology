"""
Prediction API: a two-stage gate in front of the Bio-NICE AI-integration
model.

Stage 1 (models/model1_relevance.joblib): TF-IDF + LogisticRegression,
predicts Core / Adjacent / Company-context only / Exclude - the same
relevance axis the labeling work in this project produced by hand. If this
predicts "Exclude", the input is treated as out-of-scope (e.g. "soccer
player") and the API returns a rejection message without running Model 2 at
all - a confident-but-meaningless AI-integration score is worse than no
score.

Stage 2 (models/model2_ai_integration.joblib): only runs for inputs Model 1
did NOT call Exclude. Predicts None / Low / Moderate (never High - see note
below), exactly as before this gate was added.

Design choice made deliberately, not by default: "Company-context only"
(e.g. an HR or Sales role at a real biotech company) still passes through to
Model 2, which already maps it to a meaningful "None" AI-integration answer
via derive_ai_integration_level() in the training script. That's a true,
useful answer ("this real biotech-company role has no AI integration"), not
a meaningless one - the rejection gate is reserved for "Exclude" only, i.e.
postings outside the Bio-NICE domain entirely.

Feature construction for both models MUST stay byte-identical to their
respective training scripts (src/train_relevance_model.py and
src/train_ai_integration_model.py), verified directly against those scripts
and the actual training CSV:
- Training text for BOTH models = f"{title_for_agent} | {description_for_agent}"
  with runs of whitespace collapsed to a single space and stripped. Both
  scripts share the same TEXT_COLUMNS list; only title/description exist in
  this dataset, so both reduce to the same two-field join.
- Model 1 uses TF-IDF only, no structured features, no hstack.
- Model 2 uses TF-IDF + 6 structured term-hit-count features via hstack, in
  the exact order recorded in its bundle's "structured_feature_names"
  (asserted at startup).
- Both saved TfidfVectorizers are already fit; only .transform() is valid.

Known model limitation (verified, not a bug here): Model 2's classifier.classes_
is ["Low", "Moderate", "None"] - there were zero "High" examples in the
training data, so this model can never return "High" even though the label
schema defines it.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional
import re

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.sparse import hstack

MODEL1_PATH = Path(__file__).resolve().parent / "model1_relevance.joblib"
MODEL2_PATH = Path(__file__).resolve().parent / "model2_ai_integration.joblib"

RELEVANCE_REJECT_LABELS = {"Exclude"}

EXPECTED_STRUCTURED_FEATURE_ORDER = [
    "ai_ml_term_count",
    "data_science_term_count",
    "comp_bio_term_count",
    "biotech_domain_term_count",
    "tools_framework_count",
    "total_computational_signals",
]

# --- term lists copied verbatim from src/train_ai_integration_model.py ---
# Do not reorder, "clean up", or deduplicate - Model 2 was trained against
# exactly these lists via count_term_hits(). Model 1 does not use these; it
# is TF-IDF only.

AI_ML_TERMS = [
    "machine learning", "deep learning", "artificial intelligence",
    "generative ai", "large language model", "llm", "neural network",
    "natural language processing", "nlp", "computer vision",
    "reinforcement learning", "transformer", "gpt", "bert",
    "random forest", "gradient boosting", "xgboost",
    "convolutional neural", "recurrent neural", "gan",
    "diffusion model", "fine-tuning", "pre-trained",
    "foundation model", "rag", "retrieval augmented",
]

DATA_SCIENCE_TERMS = [
    "data science", "data scientist", "data engineer", "data engineering",
    "data analytics", "data analysis", "predictive modeling",
    "statistical modeling", "statistical analysis", "bayesian",
    "regression", "classification", "clustering", "dimensionality reduction",
    "feature engineering", "model development", "model validation",
    "a/b testing", "hypothesis testing", "causal inference",
]

COMPUTATIONAL_BIO_TERMS = [
    "bioinformatics", "computational biology", "genomics", "proteomics",
    "transcriptomics", "metabolomics", "single-cell", "rna-seq",
    "whole genome", "exome", "variant calling", "pathway analysis",
    "molecular dynamics", "molecular modeling", "docking",
    "computational chemistry", "cheminformatics", "qsar",
    "systems biology", "network biology", "structural biology",
    "protein structure prediction", "alphafold",
]

BIOTECH_DOMAIN_TERMS = [
    "drug discovery", "target identification", "lead optimization",
    "clinical trial", "clinical data", "pharmacovigilance",
    "biostatistics", "biostatistician", "statistical programming",
    "pharmacometrics", "qsp", "pk/pd", "pharmacokinetics",
    "real world evidence", "real-world evidence", "rwe", "heor",
    "biomarker", "companion diagnostic", "precision medicine",
    "cell therapy", "gene therapy", "crispr", "antibody",
    "protein engineering", "biologics",
]

TOOLS_FRAMEWORKS = [
    "python", "r programming", "sas", "stata",
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
    "pandas", "numpy", "scipy", "spark", "hadoop",
    "aws", "azure", "gcp", "docker", "kubernetes",
    "sql", "nosql", "mongodb", "snowflake", "databricks",
    "tableau", "power bi", "jupyter",
]


def count_term_hits(text: str, terms: list) -> int:
    # Matches count_term_hits() in the training scripts exactly: a 0/1
    # containment check per term, not an occurrence count.
    text = " " + str(text).lower() + " "
    return sum(1 for t in terms if t in text)


def build_model_text(title: Optional[str], description: Optional[str]) -> str:
    # Shared by both models - both training scripts' TEXT_COLUMNS reduce to
    # the same title+description join on this dataset (verified directly).
    title = "" if title is None else str(title)
    description = "" if description is None else str(description)
    joined = f"{title} | {description}"
    return re.sub(r"\s+", " ", joined).strip()


def build_structured_features(text: str) -> np.ndarray:
    ai_ml = count_term_hits(text, AI_ML_TERMS)
    data_science = count_term_hits(text, DATA_SCIENCE_TERMS)
    comp_bio = count_term_hits(text, COMPUTATIONAL_BIO_TERMS)
    biotech_domain = count_term_hits(text, BIOTECH_DOMAIN_TERMS)
    tools = count_term_hits(text, TOOLS_FRAMEWORKS)
    # matches training: total excludes biotech_domain_term_count
    total = ai_ml + data_science + comp_bio + tools
    return np.array(
        [[ai_ml, data_science, comp_bio, biotech_domain, tools, total]],
        dtype=np.float64,
    )


_bundle: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # joblib.load() below deserializes pickles. This is safe here: both
    # files are our own model artifacts, trained locally in this repo
    # (src/train_relevance_model.py, src/train_ai_integration_model.py) and
    # committed alongside this API, not files accepted from an untrusted
    # source or user upload.
    _bundle["model1"] = joblib.load(MODEL1_PATH)
    _bundle["model2"] = joblib.load(MODEL2_PATH)

    actual_order = _bundle["model2"]["structured_feature_names"]
    if actual_order != EXPECTED_STRUCTURED_FEATURE_ORDER:
        raise RuntimeError(
            "structured_feature_names in the saved Model 2 no longer matches "
            f"the order this API assumes. Saved: {actual_order}, "
            f"expected: {EXPECTED_STRUCTURED_FEATURE_ORDER}. Update app.py "
            "before serving predictions."
        )
    yield
    _bundle.clear()


app = FastAPI(title="Bio-NICE Relevance + AI Integration Predictor", lifespan=lifespan)

# CORS wide open for now - lock allow_origins down to the Vercel domain once
# it's known.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    title: str = ""
    description: str = ""


class PredictResponse(BaseModel):
    relevant: bool
    relevance_prediction: str
    relevance_probabilities: Optional[Dict[str, float]] = None
    message: Optional[str] = None
    prediction: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(_bundle)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not _bundle:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    text = build_model_text(req.title, req.description)

    # --- Stage 1: relevance gate ---
    model1 = _bundle["model1"]
    vectorizer1 = model1["vectorizer"]
    classifier1 = model1["classifier"]

    X1 = vectorizer1.transform([text])
    relevance_pred = classifier1.predict(X1)[0]

    relevance_probabilities = None
    if hasattr(classifier1, "predict_proba"):
        proba1 = classifier1.predict_proba(X1)[0]
        relevance_probabilities = {
            cls: float(p) for cls, p in zip(classifier1.classes_, proba1)
        }

    if relevance_pred in RELEVANCE_REJECT_LABELS:
        return PredictResponse(
            relevant=False,
            relevance_prediction=relevance_pred,
            relevance_probabilities=relevance_probabilities,
            message=(
                "This doesn't look like a Bio-NICE-relevant posting "
                f"(relevance model classified it as '{relevance_pred}'), so no "
                "AI-integration prediction is available - that score would not "
                "be meaningful for an out-of-scope input."
            ),
        )

    # --- Stage 2: AI-integration prediction ---
    model2 = _bundle["model2"]
    vectorizer2 = model2["vectorizer"]
    classifier2 = model2["classifier"]

    X2_tfidf = vectorizer2.transform([text])
    X2_structured = build_structured_features(text)
    X2 = hstack([X2_tfidf, X2_structured]).tocsr()

    prediction = classifier2.predict(X2)[0]

    probabilities = None
    if hasattr(classifier2, "predict_proba"):
        proba2 = classifier2.predict_proba(X2)[0]
        probabilities = {cls: float(p) for cls, p in zip(classifier2.classes_, proba2)}

    return PredictResponse(
        relevant=True,
        relevance_prediction=relevance_pred,
        relevance_probabilities=relevance_probabilities,
        prediction=prediction,
        probabilities=probabilities,
    )
