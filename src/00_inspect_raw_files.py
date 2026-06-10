from pathlib import Path
import pandas as pd

RAW_DIR = Path(r"D:\P\BioNICE_Methodology\data\raw")

for folder in RAW_DIR.iterdir():
    if folder.is_dir():
        print(f"\n==============================")
        print(f"Folder: {folder.name}")
        print(f"==============================")

        files = list(folder.rglob("*"))
        for file in files:
            if file.is_file():
                print(f"File: {file.name}")

                if file.suffix.lower() == ".csv":
                    try:
                        df = pd.read_csv(file, nrows=5)
                        print(f"  Shape preview: {df.shape}")
                        print(f"  Columns: {df.columns.tolist()}")
                    except Exception as e:
                        print(f"  Could not read CSV: {e}")

                elif file.suffix.lower() in [".xlsx", ".xls"]:
                    try:
                        df = pd.read_excel(file, nrows=5)
                        print(f"  Shape preview: {df.shape}")
                        print(f"  Columns: {df.columns.tolist()}")
                    except Exception as e:
                        print(f"  Could not read Excel: {e}")