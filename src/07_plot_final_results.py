from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

TABLES = Path(r"D:\P\BioNICE_Methodology\outputs\tables")
FIGS = Path(r"D:\P\BioNICE_Methodology\outputs\figures")
FIGS.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Plot 1: Final label distribution
# -----------------------------
label_dist = pd.read_csv(TABLES / "final_label_distribution.csv")

plt.figure(figsize=(8, 5))
plt.bar(label_dist["label"], label_dist["count"])
plt.title("Final Bio-NICE Label Distribution")
plt.xlabel("Label")
plt.ylabel("Number of job postings")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(FIGS / "final_label_distribution.png", dpi=300)
plt.close()

# -----------------------------
# Plot 2: Core role distribution
# -----------------------------
core_role = pd.read_csv(TABLES / "core_role_distribution.csv")
core_role = core_role.sort_values("count", ascending=True)

plt.figure(figsize=(9, 6))
plt.barh(core_role["core_role"], core_role["count"])
plt.title("Core Bio-NICE Role Distribution")
plt.xlabel("Number of Core job postings")
plt.ylabel("Core role")
plt.tight_layout()
plt.savefig(FIGS / "core_role_distribution.png", dpi=300)
plt.close()

# -----------------------------
# Plot 3: Adjacent role distribution
# -----------------------------
adjacent_role = pd.read_csv(TABLES / "adjacent_role_distribution.csv")
adjacent_role = adjacent_role.sort_values("count", ascending=True)

plt.figure(figsize=(9, 6))
plt.barh(adjacent_role["adjacent_role"], adjacent_role["count"])
plt.title("Adjacent Bio-NICE Role Distribution")
plt.xlabel("Number of Adjacent job postings")
plt.ylabel("Adjacent role")
plt.tight_layout()
plt.savefig(FIGS / "adjacent_role_distribution.png", dpi=300)
plt.close()

# -----------------------------
# Plot 4: Intensity distribution
# -----------------------------
intensity = pd.read_csv(TABLES / "final_intensity_distribution.csv")
intensity = intensity.sort_values("intensity")

plt.figure(figsize=(7, 5))
plt.bar(intensity["intensity"].astype(str), intensity["count"])
plt.title("Bio-NICE AI/Data Intensity Distribution")
plt.xlabel("AI/Data intensity")
plt.ylabel("Number of job postings")
plt.tight_layout()
plt.savefig(FIGS / "final_intensity_distribution.png", dpi=300)
plt.close()

print("Saved figures to:")
print(FIGS)