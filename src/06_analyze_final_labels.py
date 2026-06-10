from pathlib import Path
import pandas as pd
import numpy as np

IN = Path(r"D:\P\BioNICE_Methodology\data\processed\linkedin_v2_ai_labeled_broad_biotech_pool_agent_loop_cleaned_v1.csv")

OUT_TABLES = Path(r"D:\P\BioNICE_Methodology\outputs\tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

print("Rows:", len(df))
print("Columns:", len(df.columns))

# -----------------------------
# Normalize final columns
# -----------------------------
final_label = "agent_loop_clean_label"
final_role = "agent_loop_clean_role"
final_intensity = "agent_loop_clean_intensity"

for col in [final_label, final_role, final_intensity, "agent_loop_confidence", "agent_loop_needs_human_review"]:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

# Fix intensity format
df[final_intensity] = df[final_intensity].replace({
    "0.0": "0",
    "1.0": "1",
    "2.0": "2",
    "3.0": "3",
})

# -----------------------------
# 1. Overall label distribution
# -----------------------------
label_dist = df[final_label].value_counts().reset_index()
label_dist.columns = ["label", "count"]
label_dist["percent"] = label_dist["count"] / len(df) * 100

print("\n=== Final label distribution ===")
print(label_dist.to_string(index=False))

label_dist.to_csv(OUT_TABLES / "final_label_distribution.csv", index=False)

# -----------------------------
# 2. Intensity distribution
# -----------------------------
intensity_dist = df[final_intensity].value_counts().reset_index()
intensity_dist.columns = ["intensity", "count"]
intensity_dist["percent"] = intensity_dist["count"] / len(df) * 100

print("\n=== Final intensity distribution ===")
print(intensity_dist.to_string(index=False))

intensity_dist.to_csv(OUT_TABLES / "final_intensity_distribution.csv", index=False)

# -----------------------------
# 3. Role distribution
# -----------------------------
role_dist = df[final_role].value_counts().reset_index()
role_dist.columns = ["role", "count"]
role_dist["percent"] = role_dist["count"] / len(df) * 100

print("\n=== Final role distribution ===")
print(role_dist.to_string(index=False))

role_dist.to_csv(OUT_TABLES / "final_role_distribution.csv", index=False)

# -----------------------------
# 4. Core role distribution
# -----------------------------
core = df[df[final_label] == "Core"].copy()

core_role_dist = core[final_role].value_counts().reset_index()
core_role_dist.columns = ["core_role", "count"]
core_role_dist["percent_of_core"] = core_role_dist["count"] / len(core) * 100

print("\n=== Core role distribution ===")
print(core_role_dist.to_string(index=False))

core_role_dist.to_csv(OUT_TABLES / "core_role_distribution.csv", index=False)

# -----------------------------
# 5. Adjacent role distribution
# -----------------------------
adjacent = df[df[final_label] == "Adjacent"].copy()

adjacent_role_dist = adjacent[final_role].value_counts().reset_index()
adjacent_role_dist.columns = ["adjacent_role", "count"]
adjacent_role_dist["percent_of_adjacent"] = adjacent_role_dist["count"] / len(adjacent) * 100

print("\n=== Adjacent role distribution ===")
print(adjacent_role_dist.to_string(index=False))

adjacent_role_dist.to_csv(OUT_TABLES / "adjacent_role_distribution.csv", index=False)

# -----------------------------
# 6. Confidence distribution
# -----------------------------
if "agent_loop_confidence" in df.columns:
    conf_dist = df["agent_loop_confidence"].value_counts().reset_index()
    conf_dist.columns = ["confidence", "count"]
    conf_dist["percent"] = conf_dist["count"] / len(df) * 100

    print("\n=== Confidence distribution ===")
    print(conf_dist.to_string(index=False))

    conf_dist.to_csv(OUT_TABLES / "final_confidence_distribution.csv", index=False)

# -----------------------------
# 7. Human review distribution
# -----------------------------
if "agent_loop_needs_human_review" in df.columns:
    review_dist = df["agent_loop_needs_human_review"].value_counts().reset_index()
    review_dist.columns = ["needs_human_review", "count"]
    review_dist["percent"] = review_dist["count"] / len(df) * 100

    print("\n=== Human review flag distribution ===")
    print(review_dist.to_string(index=False))

    review_dist.to_csv(OUT_TABLES / "human_review_flag_distribution.csv", index=False)

# -----------------------------
# 8. Label by confidence
# -----------------------------
if "agent_loop_confidence" in df.columns:
    label_by_conf = pd.crosstab(
        df[final_label],
        df["agent_loop_confidence"],
        margins=True
    )

    print("\n=== Label by confidence ===")
    print(label_by_conf)

    label_by_conf.to_csv(OUT_TABLES / "label_by_confidence.csv")

# -----------------------------
# 9. Label by intensity
# -----------------------------
label_by_intensity = pd.crosstab(
    df[final_label],
    df[final_intensity],
    margins=True
)

print("\n=== Label by intensity ===")
print(label_by_intensity)

label_by_intensity.to_csv(OUT_TABLES / "label_by_intensity.csv")

# -----------------------------
# 10. Core intensity by role
# -----------------------------
core_role_intensity = pd.crosstab(
    core[final_role],
    core[final_intensity],
    margins=True
)

print("\n=== Core role by intensity ===")
print(core_role_intensity)

core_role_intensity.to_csv(OUT_TABLES / "core_role_by_intensity.csv")

# -----------------------------
# 11. Top companies among Core jobs
# -----------------------------
if "company_name" in df.columns:
    core_company = core["company_name"].fillna("").value_counts().reset_index()
    core_company.columns = ["company_name", "core_count"]
    core_company["percent_of_core"] = core_company["core_count"] / len(core) * 100

    print("\n=== Top companies among Core jobs ===")
    print(core_company.head(30).to_string(index=False))

    core_company.to_csv(OUT_TABLES / "top_core_companies.csv", index=False)

# -----------------------------
# 12. Top titles among Core jobs
# -----------------------------
if "title_for_agent" in df.columns:
    core_titles = core["title_for_agent"].fillna("").value_counts().reset_index()
    core_titles.columns = ["title", "core_count"]

    print("\n=== Top Core titles ===")
    print(core_titles.head(50).to_string(index=False))

    core_titles.to_csv(OUT_TABLES / "top_core_titles.csv", index=False)

# -----------------------------
# 13. Core by job industry
# -----------------------------
if "job_industry_names" in df.columns:
    core_industry = core["job_industry_names"].fillna("").value_counts().reset_index()
    core_industry.columns = ["job_industry_names", "core_count"]
    core_industry["percent_of_core"] = core_industry["core_count"] / len(core) * 100

    print("\n=== Core by job industry ===")
    print(core_industry.head(40).to_string(index=False))

    core_industry.to_csv(OUT_TABLES / "core_by_job_industry.csv", index=False)

# -----------------------------
# 14. Save main Core subset
# -----------------------------
core_out = Path(r"D:\P\BioNICE_Methodology\data\processed\final_core_bionice_jobs.csv")
core.to_csv(core_out, index=False, encoding="utf-8-sig")

print("\nSaved Core subset:")
print(core_out)

print("\nSaved output tables to:")
print(OUT_TABLES)