from pathlib import Path
import pandas as pd

TABLES = Path(r"D:\P\BioNICE_Methodology\outputs\tables")
OUT = Path(r"D:\P\BioNICE_Methodology\outputs")
OUT.mkdir(parents=True, exist_ok=True)

label_dist = pd.read_csv(TABLES / "final_label_distribution.csv")
role_dist = pd.read_csv(TABLES / "final_role_distribution.csv")
core_role = pd.read_csv(TABLES / "core_role_distribution.csv")
intensity = pd.read_csv(TABLES / "final_intensity_distribution.csv")

total = label_dist["count"].sum()
core_n = int(label_dist.loc[label_dist["label"] == "Core", "count"].iloc[0])
adj_n = int(label_dist.loc[label_dist["label"] == "Adjacent", "count"].iloc[0])
company_n = int(label_dist.loc[label_dist["label"] == "Company-context only", "count"].iloc[0])
exclude_n = int(label_dist.loc[label_dist["label"] == "Exclude", "count"].iloc[0])

core_pct = core_n / total * 100
adj_pct = adj_n / total * 100
company_pct = company_n / total * 100
exclude_pct = exclude_n / total * 100

top_core_roles = core_role.head(10).copy()

summary_text = f"""
Bio-NICE Pilot Final Labeling Summary
====================================

Input dataset:
- Broad biotech/life-science candidate pool: {total:,} job postings

Final label distribution:
- Core: {core_n:,} ({core_pct:.1f}%)
- Adjacent: {adj_n:,} ({adj_pct:.1f}%)
- Company-context only: {company_n:,} ({company_pct:.1f}%)
- Exclude: {exclude_n:,} ({exclude_pct:.1f}%)

Interpretation:
Among the broad biotech/life-science candidate postings, {core_n:,} postings were classified as Core Bio-NICE roles after evidence-first AI-assisted labeling and rule-based quality control. Core roles required explicit job-level evidence of data science, AI/ML, computational biology, bioinformatics, biostatistics, statistical programming, clinical data management, RWE/HEOR, QSP/pharmacometrics, computational chemistry, omics/biomarker analytics, or related computational biomedical work.

Adjacent roles represented life-science or biomedical jobs without strong AI/data/computational evidence, including wet-lab research, QA/GxP, biomanufacturing, clinical operations, regulatory, and field application roles.

Company-context-only roles were business, sales, finance, HR, legal, marketing, commercial, or generic operations jobs at relevant biotech/pharma/healthcare/AI companies.

Excluded roles were out-of-scope patient-care, technician, service, warehouse, retail, maintenance, or unrelated jobs.

Top Core Bio-NICE roles:
"""

for _, row in top_core_roles.iterrows():
    summary_text += f"- {row['core_role']}: {int(row['count']):,} ({row['percent_of_core']:.1f}% of Core)\\n"

out_path = OUT / "final_bionice_labeling_summary.txt"
out_path.write_text(summary_text, encoding="utf-8")

print(summary_text)
print("\nSaved summary:")
print(out_path)