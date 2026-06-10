from pathlib import Path
import pandas as pd

IN = Path(r"D:\P\BioNICE_Methodology\data\processed\linkedin_v2_ai_labeled_broad_biotech_pool_merged_qc_agent_loop.csv")
OUT = Path(r"D:\P\BioNICE_Methodology\data\processed")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

print("Rows:", len(df))
print("Columns:")
print(df.columns.tolist())

# ---- auto-detect column names ----
title_col = "title_for_agent" if "title_for_agent" in df.columns else "title"
desc_col = "description_for_agent" if "description_for_agent" in df.columns else "description"

label_candidates = [
    "agent_loop_final_label",
    "agent_qc_final_label",
    "final_label",
    "qc_final_label",
]

intensity_candidates = [
    "agent_loop_final_intensity",
    "agent_qc_final_intensity",
    "final_intensity",
    "qc_final_intensity",
]

role_candidates = [
    "agent_loop_final_role",
    "agent_qc_final_role",
    "final_role",
    "qc_final_role",
]

reason_candidates = [
    "qc_decision_reason",
    "agent_qc_notes",
    "agent_loop_notes",
    "qc_notes",
    "agent_qc_final_notes",
]

def pick_col(candidates, df):
    for c in candidates:
        if c in df.columns:
            return c
    return None

label_col = pick_col(label_candidates, df)
intensity_col = pick_col(intensity_candidates, df)
role_col = pick_col(role_candidates, df)
reason_col = pick_col(reason_candidates, df)

print("\nDetected columns:")
print("label_col:", label_col)
print("intensity_col:", intensity_col)
print("role_col:", role_col)
print("reason_col:", reason_col)

if label_col is None or intensity_col is None or role_col is None:
    raise ValueError("Could not detect final label/intensity/role columns. Check printed column names above.")

if reason_col is None:
    df["temporary_reason_col"] = ""
    reason_col = "temporary_reason_col"

# Normalize required fields
for col in [title_col, desc_col, label_col, intensity_col, role_col, reason_col]:
    df[col] = df[col].fillna("").astype(str)

# Create cleaned columns
df["agent_loop_clean_label"] = df[label_col]
df["agent_loop_clean_intensity"] = df[intensity_col].astype(str).str.replace(".0", "", regex=False)
df["agent_loop_clean_role"] = df[role_col]
df["agent_loop_clean_notes"] = df[reason_col]

df["title_lower"] = df[title_col].str.lower()
df["desc_lower"] = df[desc_col].str.lower()
df["text_lower"] = df["title_lower"] + " " + df["desc_lower"]

def has_any(text, terms):
    text = " " + str(text).lower() + " "
    return any(t in text for t in terms)

strong_core_terms = [
    "bioinformatic", "computational biology", "data scientist",
    "machine learning", "deep learning", "artificial intelligence",
    "biostatistic", "statistical programmer", "sas programmer",
    "clinical data manager", "clinical data management",
    "clinical data scientist", "clinical informatics",
    "rwe", "real world evidence", "real-world evidence",
    "heor", "outcomes research", "claims data",
    "qsp", "pharmacometrics", "pk/pd",
    "genomics", "omics", "rna-seq", "single-cell",
    "computational chemistry", "cheminformatics",
    "drug discovery", "protein design", "molecular modeling",
    "model development", "predictive modeling"
]

business_core_false_positive_terms = [
    "accounting", "tax", "finance", "financial", "fp&a",
    "hr ", "human resources", "talent management", "recruitment",
    "revenue operations", "sales operations", "commercial operations",
    "export supply", "supply operations", "procurement",
    "enterprise resources planning", "erp",
    "project manager", "program manager",
    "operations manager", "operations analyst",
    "operations specialist",
    "administrative", "assistant",
]

exclude_terms = [
    "biomedical technician", "medical assistant",
    "pharmacy technician", "nurse", "physician", "dentist",
    "packer", "machine operator", "warehouse", "driver",
    "security", "maintenance", "pest control", "electrician"
]

df["has_strong_core_evidence"] = df["text_lower"].apply(lambda x: has_any(x, strong_core_terms))
df["business_fp_title"] = df["title_lower"].apply(lambda x: has_any(x, business_core_false_positive_terms))
df["exclude_fp_title"] = df["title_lower"].apply(lambda x: has_any(x, exclude_terms))

# Business/admin/ops Core without strong Core evidence -> Company-context only
mask_business_fp = (
    (df[label_col] == "Core")
    & df["business_fp_title"]
    & ~df["has_strong_core_evidence"]
)

df.loc[mask_business_fp, "agent_loop_clean_label"] = "Company-context only"
df.loc[mask_business_fp, "agent_loop_clean_intensity"] = "0"
df.loc[mask_business_fp, "agent_loop_clean_role"] = "Company-context only"
df.loc[mask_business_fp, "agent_loop_clean_notes"] = (
    "Post-QC: business/admin/operations role without explicit Bio-NICE Core evidence."
)

# Exclude-type Core without strong Core evidence -> Exclude
mask_exclude_fp = (
    (df[label_col] == "Core")
    & df["exclude_fp_title"]
    & ~df["has_strong_core_evidence"]
)

df.loc[mask_exclude_fp, "agent_loop_clean_label"] = "Exclude"
df.loc[mask_exclude_fp, "agent_loop_clean_intensity"] = "0"
df.loc[mask_exclude_fp, "agent_loop_clean_role"] = "Exclude"
df.loc[mask_exclude_fp, "agent_loop_clean_notes"] = (
    "Post-QC: patient-care/service/technician role without explicit Bio-NICE Core evidence."
)

out_path = OUT / "linkedin_v2_ai_labeled_broad_biotech_pool_agent_loop_cleaned_v1.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print("\nSaved:")
print(out_path)

print("\nOriginal detected label distribution:")
print(df[label_col].value_counts())

print("\nCleaned agent_loop label distribution:")
print(df["agent_loop_clean_label"].value_counts())

print("\nOriginal Core:", (df[label_col] == "Core").sum())
print("Cleaned Core:", (df["agent_loop_clean_label"] == "Core").sum())
print("Business false Core corrected:", mask_business_fp.sum())
print("Exclude false Core corrected:", mask_exclude_fp.sum())

print("\nCleaned role distribution:")
print(df["agent_loop_clean_role"].value_counts().head(30))