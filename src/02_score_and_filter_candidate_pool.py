from pathlib import Path
import pandas as pd

IN = Path(r"D:\P\BioNICE_Methodology\data\processed\linkedin_merged_v2_with_company_context.csv")
OUT = Path(r"D:\P\BioNICE_Methodology\data\processed")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

for col in [
    "title", "description", "skills_desc", "job_skill_names",
    "job_industry_names", "company_description",
    "company_industry_names", "company_specialities"
]:
    if col not in df.columns:
        df[col] = ""

df["title_text"] = df["title"].fillna("").astype(str).str.lower()
df["job_text"] = (
    df["title"].fillna("").astype(str) + " " +
    df["description"].fillna("").astype(str) + " " +
    df["skills_desc"].fillna("").astype(str) + " " +
    df["job_skill_names"].fillna("").astype(str)
).str.lower()

df["job_industry_text"] = df["job_industry_names"].fillna("").astype(str).str.lower()

df["company_text"] = (
    df["company_description"].fillna("").astype(str) + " " +
    df["company_industry_names"].fillna("").astype(str) + " " +
    df["company_specialities"].fillna("").astype(str)
).str.lower()


# Strong biotech/life-science industry signal
strong_bio_industry_terms = [
    "biotechnology research",
    "biotechnology",
    "pharmaceutical manufacturing",
    "medical equipment manufacturing",
    "medical device",
    "research services",
    "medical and diagnostic laboratories"
]

# Broader but weaker industry signal
weak_bio_industry_terms = [
    "hospitals and health care",
    "medical practices",
    "public health",
    "veterinary services",
    "chemical manufacturing",
    "higher education"
]

# Job-level biotech/domain terms
bio_job_terms = [
    "bioinformatics",
    "computational biology",
    "genomics",
    "genomic",
    "rna-seq",
    "single-cell",
    "sequencing",
    "biomarker",
    "drug discovery",
    "target discovery",
    "protein design",
    "protein engineering",
    "molecular modeling",
    "cheminformatics",
    "computational chemistry",
    "therapeutics",
    "oncology",
    "clinical trial",
    "clinical research",
    "real world evidence",
    "real-world evidence",
    "rwe",
    "qsp",
    "pharmacometrics",
    "biostatistics",
    "pharmacovigilance",
    "gmp",
    "cmc",
    "biomanufacturing",
    "diagnostics",
    "ivd"
]

# Job-level computational/data/AI terms
computational_job_terms = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "generative ai",
    "large language model",
    "llm",
    "data science",
    "data scientist",
    "data engineer",
    "bioinformatics",
    "computational biology",
    "statistical programming",
    "statistical programmer",
    "biostatistician",
    "biostatistics",
    "informatics",
    "predictive modeling",
    "modeling",
    "modelling",
    "python",
    "r programming",
    "sas",
    "pytorch",
    "tensorflow",
    "clinical data",
    "real world data",
    "real-world data",
    "rwe",
    "pharmacometrics",
    "qsp"
]

# Strong title signals
strong_title_terms = [
    "bioinformatics",
    "computational biologist",
    "computational biology",
    "data scientist",
    "machine learning",
    "statistical programmer",
    "biostatistician",
    "biostatistics",
    "data engineer",
    "clinical data",
    "rwe",
    "real world evidence",
    "qsp",
    "pharmacometrics",
    "computational chemist",
    "cheminformatics",
    "informatics",
    "biomarker",
    "genomics"
]

# Company-level supporting terms
company_bio_terms = [
    "biotechnology",
    "biotech",
    "pharmaceutical",
    "pharma",
    "biopharma",
    "life sciences",
    "life science",
    "oncology",
    "therapeutics",
    "drug discovery",
    "clinical research",
    "medical device",
    "diagnostics",
    "genomics",
    "biostatistics"
]

company_computational_terms = [
    "artificial intelligence",
    "machine learning",
    "data science",
    "data analytics",
    "ai/ml",
    "big data",
    "data engineering",
    "predictive analytics"
]

# Obvious false positives
exclude_title_terms = [
    "retail",
    "store manager",
    "assistant manager",
    "operations assistant manager",
    "front end",
    "deli associate",
    "barista",
    "cake decorator",
    "cashier",
    "warehouse",
    "material handler",
    "customer service",
    "sales associate",
    "sales representative",
    "account executive",
    "driver",
    "maintenance",
    "mechanic",
    "pharmacy technician",
    "pharmacist",
    "registered nurse",
    " rn ",
    "lpn",
    "licensed practical nurse",
    "medical assistant",
    "dental assistant",
    "patient care technician",
    "patient care coordinator",
    "phlebotomist",
    "occupational therapist",
    "physical therapist",
    "cyber threat",
    "osint"
]


def has_any(text, terms):
    text = str(text).lower()
    return any(term in text for term in terms)


def count_any(text, terms):
    text = str(text).lower()
    return sum(1 for term in terms if term in text)


# Scores
df["strong_bio_industry_count"] = df["job_industry_text"].apply(lambda x: count_any(x, strong_bio_industry_terms))
df["weak_bio_industry_count"] = df["job_industry_text"].apply(lambda x: count_any(x, weak_bio_industry_terms))
df["bio_job_count"] = df["job_text"].apply(lambda x: count_any(x, bio_job_terms))
df["computational_job_count"] = df["job_text"].apply(lambda x: count_any(x, computational_job_terms))
df["strong_title_count"] = df["title_text"].apply(lambda x: count_any(x, strong_title_terms))
df["company_bio_count"] = df["company_text"].apply(lambda x: count_any(x, company_bio_terms))
df["company_computational_count"] = df["company_text"].apply(lambda x: count_any(x, company_computational_terms))
df["exclude_title"] = df["title_text"].apply(lambda x: has_any(x, exclude_title_terms))

# Weighted scoring
df["biotech_score"] = (
    3 * df["strong_bio_industry_count"] +
    1 * df["weak_bio_industry_count"] +
    2 * df["bio_job_count"] +
    1 * df["company_bio_count"]
)

df["computational_score"] = (
    3 * df["strong_title_count"] +
    2 * df["computational_job_count"] +
    1 * df["company_computational_count"]
)

# Candidate pools
broad_biotech = df[
    (df["biotech_score"] >= 3)
    & (~df["exclude_title"])
].copy()

computational_biotech = df[
    (df["biotech_score"] >= 3)
    & (df["computational_score"] >= 3)
    & (~df["exclude_title"])
].copy()

core_bionice = df[
    (
        (df["strong_title_count"] >= 1)
        | ((df["bio_job_count"] >= 1) & (df["computational_job_count"] >= 1))
    )
    & (df["biotech_score"] >= 3)
    & (df["computational_score"] >= 3)
    & (~df["exclude_title"])
].copy()

print("Total postings:", len(df))
print("Broad biotech pool:", len(broad_biotech))
print("Computational biotech pool:", len(computational_biotech))
print("Core Bio-NICE candidate pool:", len(core_bionice))

print("\nTop core titles:")
print(core_bionice["title"].value_counts().head(80).to_string())

print("\nTop core job industries:")
print(core_bionice["job_industry_names"].value_counts().head(60).to_string())

print("\nTop core company industries:")
print(core_bionice["company_industry_names"].value_counts().head(60).to_string())

keep_cols = [
    "job_id", "company_name", "title", "description", "location",
    "formatted_experience_level", "normalized_salary",
    "original_listed_time", "listed_time", "job_posting_url",
    "job_skill_names", "job_industry_names",
    "company_description", "company_industry_names", "company_specialities",
    "biotech_score", "computational_score",
    "strong_bio_industry_count", "weak_bio_industry_count",
    "bio_job_count", "computational_job_count", "strong_title_count",
    "company_bio_count", "company_computational_count",
    "exclude_title"
]
keep_cols = [c for c in keep_cols if c in df.columns]

broad_biotech[keep_cols].to_csv(OUT / "linkedin_v2_broad_biotech_pool.csv", index=False)
computational_biotech[keep_cols].to_csv(OUT / "linkedin_v2_computational_biotech_pool.csv", index=False)
core_bionice[keep_cols].to_csv(OUT / "linkedin_v2_core_bionice_pool.csv", index=False)

print("\nSaved:")
print(OUT / "linkedin_v2_broad_biotech_pool.csv")
print(OUT / "linkedin_v2_computational_biotech_pool.csv")
print(OUT / "linkedin_v2_core_bionice_pool.csv")