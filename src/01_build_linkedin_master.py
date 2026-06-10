from pathlib import Path
import pandas as pd

RAW = Path(r"D:\P\BioNICE_Methodology\data\raw\LinkedIn job 23-24")
OUT = Path(r"D:\P\BioNICE_Methodology\data\processed")
OUT.mkdir(parents=True, exist_ok=True)

def find_file(filename):
    matches = list(RAW.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename}")
    return matches[0]

postings = pd.read_csv(find_file("postings.csv"))
companies = pd.read_csv(find_file("companies.csv"))
company_industries = pd.read_csv(find_file("company_industries.csv"))
company_specialities = pd.read_csv(find_file("company_specialities.csv"))
job_skills = pd.read_csv(find_file("job_skills.csv"))
skills = pd.read_csv(find_file("skills.csv"))
job_industries = pd.read_csv(find_file("job_industries.csv"))
industries = pd.read_csv(find_file("industries.csv"))

# Job skills
job_skills_named = job_skills.merge(skills, on="skill_abr", how="left")
skills_by_job = (
    job_skills_named
    .groupby("job_id")["skill_name"]
    .apply(lambda x: "; ".join(sorted(set(x.dropna().astype(str)))))
    .reset_index()
    .rename(columns={"skill_name": "job_skill_names"})
)

# Job industries
job_industries_named = job_industries.merge(industries, on="industry_id", how="left")
industries_by_job = (
    job_industries_named
    .groupby("job_id")["industry_name"]
    .apply(lambda x: "; ".join(sorted(set(x.dropna().astype(str)))))
    .reset_index()
    .rename(columns={"industry_name": "job_industry_names"})
)

# Company industries
company_ind_by_company = (
    company_industries
    .groupby("company_id")["industry"]
    .apply(lambda x: "; ".join(sorted(set(x.dropna().astype(str)))))
    .reset_index()
    .rename(columns={"industry": "company_industry_names"})
)

# Company specialities
company_spec_by_company = (
    company_specialities
    .groupby("company_id")["speciality"]
    .apply(lambda x: "; ".join(sorted(set(x.dropna().astype(str)))))
    .reset_index()
    .rename(columns={"speciality": "company_specialities"})
)

# Rename company fields
companies_small = companies.rename(columns={
    "name": "company_name_from_company_file",
    "description": "company_description",
    "city": "company_city",
    "state": "company_state",
    "country": "company_country"
})

# Merge all
df = postings.merge(skills_by_job, on="job_id", how="left")
df = df.merge(industries_by_job, on="job_id", how="left")
df = df.merge(companies_small, on="company_id", how="left")
df = df.merge(company_ind_by_company, on="company_id", how="left")
df = df.merge(company_spec_by_company, on="company_id", how="left")

text_cols = [
    "title",
    "description",
    "skills_desc",
    "job_skill_names",
    "job_industry_names",
    "company_description",
    "company_industry_names",
    "company_specialities"
]

for col in text_cols:
    if col not in df.columns:
        df[col] = ""

df["combined_text_full"] = (
    df[text_cols]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.lower()
)

out_path = OUT / "linkedin_merged_v2_with_company_context.csv"
df.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())