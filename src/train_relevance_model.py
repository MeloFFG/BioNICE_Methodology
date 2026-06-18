"""
Train Model 1: Bio-NICE relevance classifier

Goal:
    Classify broad biotech job postings into:
    Core / Adjacent / Company-context only / Exclude

Model:
    TF-IDF + Logistic Regression

Expected input:
    data/processed/linkedin_v2_ai_labeled_broad_biotech_pool_agent_loop_cleaned_v1.csv

Main outputs:
    outputs/model1_relevance_classification_report.csv
    outputs/model1_relevance_confusion_matrix.csv
    outputs/model1_relevance_predictions.csv
    outputs/model1_relevance_top_terms_by_class.csv
    models/model1_relevance_tfidf_logreg.joblib
"""

from pathlib import Path
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore")


# -----------------------------
# Config
# -----------------------------

DEFAULT_INPUT = "data/processed/linkedin_v2_ai_labeled_broad_biotech_pool_agent_loop_cleaned_v1.csv"

TEXT_COLUMNS = [
    "title_for_agent",
    "job_skill_names",
    "description_for_agent",
    "company_industry_names",
    "company_specialities_for_agent",
    "company_description_for_agent",
]

TARGET_COLUMN = "agent_loop_clean_label"
REVIEW_COLUMN = "agent_loop_needs_human_review"

LABEL_ORDER = [
    "Core",
    "Adjacent",
    "Company-context only",
    "Exclude",
]


# -----------------------------
# Helper functions
# -----------------------------

def make_dirs() -> None:
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)


def load_data(input_path: str, high_confidence_only: bool = True) -> pd.DataFrame:
    """Load dataset and optionally keep only rows that do not need human review."""
    df = pd.read_csv(input_path)

    missing_cols = [col for col in [TARGET_COLUMN, *TEXT_COLUMNS] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()

    # Keep rows with valid target label.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
    df = df[df[TARGET_COLUMN].isin(LABEL_ORDER)]

    # First baseline: use cleaner labels only.
    if high_confidence_only and REVIEW_COLUMN in df.columns:
        review = df[REVIEW_COLUMN].astype(str).str.lower().str.strip()
        df = df[review.isin(["no", "false", "0", "nan", "none", ""])]

    return df


def build_model_text(df: pd.DataFrame) -> pd.Series:
    """
    Combine job-level and company-context fields into one text input.

    Note:
        If you want the model to rely less on company context,
        remove company_description_for_agent or company_specialities_for_agent.
    """
    available_cols = [col for col in TEXT_COLUMNS if col in df.columns]

    for col in available_cols:
        df[col] = df[col].fillna("").astype(str)

    text = df[available_cols].agg(" | ".join, axis=1)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()

    return text


def get_top_terms_by_class(pipe: Pipeline, top_n: int = 30) -> pd.DataFrame:
    """Extract top positive TF-IDF terms for each Logistic Regression class."""
    vectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    rows = []

    for class_index, class_name in enumerate(clf.classes_):
        coefs = clf.coef_[class_index]
        top_indices = np.argsort(coefs)[::-1][:top_n]

        for rank, idx in enumerate(top_indices, start=1):
            rows.append({
                "class": class_name,
                "rank": rank,
                "term": feature_names[idx],
                "coefficient": coefs[idx],
            })

    return pd.DataFrame(rows)


def train_relevance_model(
    input_path: str,
    high_confidence_only: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    make_dirs()

    df = load_data(input_path, high_confidence_only=high_confidence_only)
    df["model_text"] = build_model_text(df)

    X = df["model_text"]
    y = df[TARGET_COLUMN]

    print("\n=== Dataset summary ===")
    print(f"Input file: {input_path}")
    print(f"Rows used: {len(df)}")
    print("\nLabel distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X,
        y,
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30000,
            min_df=2,
            max_df=0.90,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="saga",
            multi_class="multinomial",
            n_jobs=-1,
            random_state=random_state,
        )),
    ])

    print("\nTraining model...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    # -----------------------------
    # Evaluation outputs
    # -----------------------------
    report = classification_report(
        y_test,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_path = "outputs/model1_relevance_classification_report.csv"
    report_df.to_csv(report_path)

    cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
    cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)
    cm_path = "outputs/model1_relevance_confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    # Save prediction-level file for error analysis.
    pred_df = df.loc[test_idx].copy()
    pred_df["true_label"] = y_test.values
    pred_df["pred_label"] = y_pred

    for i, class_name in enumerate(pipe.named_steps["clf"].classes_):
        pred_df[f"prob_{class_name}"] = y_proba[:, i]

    pred_path = "outputs/model1_relevance_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Top terms by class.
    top_terms_df = get_top_terms_by_class(pipe, top_n=40)
    top_terms_path = "outputs/model1_relevance_top_terms_by_class.csv"
    top_terms_df.to_csv(top_terms_path, index=False)

    # Save model.
    model_path = "models/model1_relevance_tfidf_logreg.joblib"
    joblib.dump(pipe, model_path)

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, labels=LABEL_ORDER, zero_division=0))

    print("\n=== Confusion matrix ===")
    print(cm_df)

    print("\nSaved files:")
    print(f"- {report_path}")
    print(f"- {cm_path}")
    print(f"- {pred_path}")
    print(f"- {top_terms_path}")
    print(f"- {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to labeled broad biotech CSV.",
    )
    parser.add_argument(
        "--all_rows",
        action="store_true",
        help="Use all rows, including rows marked as needing human review.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split fraction.",
    )
    args = parser.parse_args()

    train_relevance_model(
        input_path=args.input,
        high_confidence_only=not args.all_rows,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
