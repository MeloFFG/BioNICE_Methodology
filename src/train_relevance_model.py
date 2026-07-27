"""
Train Model 1: Bio-NICE relevance classifier

Goal:
    Classify broad biotech job postings into:
    Core / Adjacent / Company-context only / Exclude

Model:
    TF-IDF + Logistic Regression, evaluated with stratified k-fold CV
    (mirrors train_ai_integration_model.py's approach/bundle format so
    backend/app.py can load and use both models the same way).

Expected input:
    data/processed/linkedin_v2_ai_labeled_broad_biotech_pool_agent_loop_cleaned_v1.csv

Main outputs:
    outputs/model1_relevance_classification_report.csv
    outputs/model1_relevance_confusion_matrix.csv
    outputs/model1_relevance_cv_results.csv
    outputs/model1_relevance_predictions.csv
    outputs/model1_relevance_top_terms_by_class.csv
    models/model1_relevance.joblib
"""

from pathlib import Path
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

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

    # Only the target column is strictly required. TEXT_COLUMNS are best-effort:
    # build_model_text() below uses whichever of them actually exist. On the
    # current dataset only title_for_agent/description_for_agent exist - the
    # other 4 were never populated (verified against the actual CSV header),
    # so requiring all of TEXT_COLUMNS here would hard-crash on real data.
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {TARGET_COLUMN}")

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


def get_top_terms_by_class(vectorizer: TfidfVectorizer, clf: LogisticRegression,
                            top_n: int = 40) -> pd.DataFrame:
    """Extract top positive TF-IDF terms for each Logistic Regression class."""
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


def get_model(random_state: int = 42) -> LogisticRegression:
    # logreg only: gbt/rf are tree-based and repeatedly got killed by this
    # environment's resource limits when training Model 2 at this same
    # dataset scale (see src/train_ai_integration_model.py history).
    return LogisticRegression(max_iter=3000, class_weight="balanced",
                              solver="saga", n_jobs=-1, random_state=random_state)


def train_relevance_model(
    input_path: str,
    high_confidence_only: bool = True,
    n_folds: int = 5,
    random_state: int = 42,
) -> None:
    make_dirs()

    df = load_data(input_path, high_confidence_only=high_confidence_only)
    df["model_text"] = build_model_text(df)

    X_text = df["model_text"]
    y = df[TARGET_COLUMN]

    print("\n=== Dataset summary ===")
    print(f"Input file: {input_path}")
    print(f"Rows used: {len(df)}")
    print("\nLabel distribution:")
    print(y.value_counts())

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=2,
                            max_df=0.90, sublinear_tf=True, stop_words="english")
    X = tfidf.fit_transform(X_text)

    class_counts = y.value_counts()
    min_class_count = int(class_counts.min())
    effective_folds = min(n_folds, min_class_count)

    y_pred_all = np.full(len(y), fill_value=None, dtype=object)
    cv_df = pd.DataFrame(columns=["fold", "f1_macro", "f1_weighted", "n_train", "n_test"])

    if effective_folds < 2:
        print(f"\nSkipping cross-validation: smallest class ({class_counts.idxmin()}) "
              f"has only {min_class_count} example(s), fewer than the 2 needed for CV. "
              f"Training the final model on all {len(y)} rows without CV metrics.")
    else:
        if effective_folds < n_folds:
            print(f"\nReducing folds from {n_folds} to {effective_folds}: smallest class "
                  f"({class_counts.idxmin()}) has only {min_class_count} example(s).")
        cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
        print(f"\nRunning {effective_folds}-fold stratified cross-validation...")
        fold_results = []

        for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            clf = get_model(random_state=random_state)
            clf.fit(X_train, y_train)
            y_pred_fold = clf.predict(X_test)
            y_pred_all[test_idx] = y_pred_fold
            f1_macro = f1_score(y_test, y_pred_fold, labels=LABEL_ORDER,
                                average="macro", zero_division=0)
            f1_weighted = f1_score(y_test, y_pred_fold, labels=LABEL_ORDER,
                                   average="weighted", zero_division=0)
            fold_results.append({"fold": fold_i, "f1_macro": f1_macro,
                                  "f1_weighted": f1_weighted,
                                  "n_train": len(train_idx), "n_test": len(test_idx)})
            print(f"  Fold {fold_i}: F1-macro={f1_macro:.3f}  F1-weighted={f1_weighted:.3f}")

        cv_df = pd.DataFrame(fold_results)
        cv_df.loc[len(cv_df)] = {"fold": "mean", "f1_macro": cv_df["f1_macro"].mean(),
                                  "f1_weighted": cv_df["f1_weighted"].mean(),
                                  "n_train": "", "n_test": ""}
    cv_df.to_csv("outputs/model1_relevance_cv_results.csv", index=False)

    have_cv_preds = pd.notna(y_pred_all).all()
    if have_cv_preds:
        report = classification_report(y, y_pred_all, labels=LABEL_ORDER,
                                       output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(
            "outputs/model1_relevance_classification_report.csv")

        cm = confusion_matrix(y, y_pred_all, labels=LABEL_ORDER)
        cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)
        cm_df.to_csv("outputs/model1_relevance_confusion_matrix.csv")
    else:
        cm_df = pd.DataFrame()

    pred_df = df.copy()
    pred_df["predicted_label"] = y_pred_all
    pred_df["true_label"] = y.values
    pred_cols = ["job_id", "title_for_agent", TARGET_COLUMN,
                 "true_label", "predicted_label"]
    pred_cols = [c for c in pred_cols if c in pred_df.columns]
    pred_df[pred_cols].to_csv("outputs/model1_relevance_predictions.csv", index=False)

    print("\nTraining final model on all data...")
    final_clf = get_model(random_state=random_state)
    final_clf.fit(X, y)

    top_terms_df = get_top_terms_by_class(tfidf, final_clf, top_n=40)
    top_terms_df.to_csv("outputs/model1_relevance_top_terms_by_class.csv", index=False)

    joblib.dump({"vectorizer": tfidf, "classifier": final_clf},
                "models/model1_relevance.joblib")

    if have_cv_preds:
        print(f"\n=== Classification Report ({effective_folds}-fold CV) ===")
        print(classification_report(y, y_pred_all, labels=LABEL_ORDER, zero_division=0))
        print("=== Confusion Matrix ===")
        print(cm_df)
        print(f"\nMean F1-macro: {cv_df.iloc[-1]['f1_macro']:.3f}")
        print(f"Mean F1-weighted: {cv_df.iloc[-1]['f1_weighted']:.3f}")
    else:
        print("\nNo CV metrics available (see skip message above). "
              "Final model was still trained and saved on all available rows.")

    print("\nSaved files:")
    print("- outputs/model1_relevance_cv_results.csv")
    if have_cv_preds:
        print("- outputs/model1_relevance_classification_report.csv")
        print("- outputs/model1_relevance_confusion_matrix.csv")
    print("- outputs/model1_relevance_predictions.csv")
    print("- outputs/model1_relevance_top_terms_by_class.csv")
    print("- models/model1_relevance.joblib")


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
        "--folds",
        type=int,
        default=5,
        help="Number of stratified CV folds (auto-reduced if a class is smaller).",
    )
    args = parser.parse_args()

    train_relevance_model(
        input_path=args.input,
        high_confidence_only=not args.all_rows,
        n_folds=args.folds,
    )


if __name__ == "__main__":
    main()
