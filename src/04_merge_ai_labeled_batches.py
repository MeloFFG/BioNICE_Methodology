from pathlib import Path
import pandas as pd

# Folder containing AI-processed / labeled batch CSV files
BATCH_DIR = Path(r"D:\P\BioNICE_Methodology\data\processed\Processed batches")

# Output folder
OUT = Path(r"D:\P\BioNICE_Methodology\data\processed")
OUT.mkdir(parents=True, exist_ok=True)

# Find all CSV files in the processed batch folder
batch_files = sorted(BATCH_DIR.glob("*.csv"))

print("Batch folder:", BATCH_DIR)
print("CSV files found:", len(batch_files))

if not batch_files:
    raise FileNotFoundError(f"No CSV files found in {BATCH_DIR}")

for f in batch_files[:10]:
    print("Example file:", f.name)

dfs = []

for file in batch_files:
    try:
        df = pd.read_csv(file)
        df["source_batch_file"] = file.name
        dfs.append(df)
        print(f"Loaded: {file.name} | Rows: {len(df)} | Columns: {len(df.columns)}")
    except Exception as e:
        print(f"Could not read {file.name}: {e}")

merged = pd.concat(dfs, ignore_index=True)

print("\nMerged shape before duplicate check:", merged.shape)

# Check duplicate job_id
if "job_id" in merged.columns:
    dup_job_id = merged["job_id"].duplicated().sum()
    print("Duplicated job_id:", dup_job_id)
else:
    print("Warning: job_id column not found.")

# Check duplicate agent_row_id
if "agent_row_id" in merged.columns:
    dup_agent_row_id = merged["agent_row_id"].duplicated().sum()
    print("Duplicated agent_row_id:", dup_agent_row_id)
else:
    print("Warning: agent_row_id column not found.")

# If duplicates exist, keep the last version
if "job_id" in merged.columns:
    merged = merged.drop_duplicates(subset=["job_id"], keep="last").copy()

print("Merged shape after duplicate removal:", merged.shape)

# Save merged file
out_path = OUT / "linkedin_v2_ai_labeled_broad_biotech_pool_merged.csv"
merged.to_csv(out_path, index=False, encoding="utf-8-sig")

print("\nSaved merged labeled file:")
print(out_path)

# Quick label distribution
if "ai_manual_label" in merged.columns:
    print("\nAI manual label distribution:")
    print(merged["ai_manual_label"].value_counts(dropna=False))

if "ai_bionice_role" in merged.columns:
    print("\nTop Bio-NICE roles:")
    print(merged["ai_bionice_role"].value_counts(dropna=False).head(20))

if "ai_confidence" in merged.columns:
    print("\nAI confidence distribution:")
    print(merged["ai_confidence"].value_counts(dropna=False))