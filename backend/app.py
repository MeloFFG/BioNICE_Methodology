"""
Prediction API for models/model2_ai_integration.joblib (Bio-NICE AI-integration
level classifier: None / Low / Moderate).

Feature construction here MUST stay byte-identical to the pipeline in
src/train_ai_integration_model.py, verified directly against that script and
against the actual training CSV (data/processed/linkedin_v2_ai_labeled_broad_biotech_pool_agent_loop_cleaned_v1.csv):

- Training text = f"{title_for_agent} | {description_for_agent}" with runs of
  whitespace collapsed to a single space and stripped. The other 4 possible
  TEXT_COLUMNS (job_skill_names, company_industry_names,
  company_specialities_for_agent, company_description_for_agent) never
  existed in this dataset, so they contribute nothing at training time either
  - title+description alone reproduces training-time text exactly.
- The saved TfidfVectorizer is already fit; only .transform() is valid here.
- Structured features are 6 term-hit counts, in the exact column order
  recorded in the joblib bundle's "structured_feature_names" (asserted at
  startup). total_computational_signals sums only ai_ml + data_science +
  comp_bio + tools_framework - biotech_domain_term_count is NOT included,
  matching build_structured_features() in the training script exactly.
- hstack order is [tfidf_matrix, structured_matrix], matching training.

Known model limitation (verified, not a bug here): classifier.classes_ is
["Low", "Moderate", "None"] - there were zero "High" examples in the training
data, so this model can never return "High" even though the label schema
defines it.
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

# joblib.load() below deserializes a pickle. This is safe here: the file is
# our own model artifact, trained locally in this repo
# (src/train_ai_integration_model.py) and committed alongside this API, not
# a file accepted from an untrusted source or user upload.
MODEL_PATH = Path(__file__).resolve().parent / "model2_ai_integration.joblib"

EXPECTED_STRUCTURED_FEATURE_ORDER = [
    "ai_ml_term_count",
    "data_science_term_count",
    "comp_bio_term_count",
    "biotech_domain_term_count",
    "tools_framework_count",
    "total_computational_signals",
]

# --- term lists copied verbatim from src/train_ai_integration_model.py ---
# Do not reorder, "clean up", or deduplicate - the model was trained against
# exactly these lists via count_term_hits().

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
    # Matches count_term_hits() in the training script exactly: a 0/1
    # containment check per term, not an occurrence count.
    text = " " + str(text).lower() + " "
    return sum(1 for t in terms if t in text)


def build_model_text(title: Optional[str], description: Optional[str]) -> str:
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
    _bundle["data"] = joblib.load(MODEL_PATH)
    actual_order = _bundle["data"]["structured_feature_names"]
    if actual_order != EXPECTED_STRUCTURED_FEATURE_ORDER:
        raise RuntimeError(
            "structured_feature_names in the saved model no longer matches "
            f"the order this API assumes. Saved: {actual_order}, "
            f"expected: {EXPECTED_STRUCTURED_FEATURE_ORDER}. Update app.py "
            "before serving predictions."
        )
    yield
    _bundle.clear()


app = FastAPI(title="Bio-NICE AI Integration Predictor", lifespan=lifespan)

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
    prediction: str
    probabilities: Optional[Dict[str, float]] = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(_bundle)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not _bundle:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    vectorizer = _bundle["data"]["vectorizer"]
    classifier = _bundle["data"]["classifier"]

    text = build_model_text(req.title, req.description)
    X_tfidf = vectorizer.transform([text])
    X_structured = build_structured_features(text)
    X = hstack([X_tfidf, X_structured]).tocsr()

    prediction = classifier.predict(X)[0]

    probabilities = None
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(X)[0]
        probabilities = {cls: float(p) for cls, p in zip(classifier.classes_, proba)}

    return PredictResponse(prediction=prediction, probabilities=probabilities)
