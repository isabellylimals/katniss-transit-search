import os
import pandas as pd
import numpy as np

metadata_path = "./data/processed/metadata.csv"
koi_template_path = "./data/raw/koi_template.csv"
output_path = "./data/processed/dataset_final.csv"

# Load files
meta = pd.read_csv(metadata_path)
koi = pd.read_csv(koi_template_path, comment="#", engine="python")
koi.columns = [c.lower().strip() for c in koi.columns]

# Identify relevant columns
id_col = next((c for c in koi.columns if "kepid" in c), "kepid")
label_col = next((c for c in koi.columns if "disposition" in c), "koi_disposition")
period_col_nasa = next((c for c in koi.columns if "period" in c), "koi_period")

# Merge datasets
merged = pd.merge(
    meta,
    koi[[id_col, label_col, period_col_nasa]].rename(columns={id_col: "kic_id"}),
    on="kic_id",
    how="inner"
)

# Period consistency check
merged["period_diff"] = np.abs(merged["period_days"] - merged[period_col_nasa])
merged["relative_diff"] = merged["period_diff"] / merged["period_days"]

# Keep best match per KOI
merged = merged.sort_values("relative_diff")
merged = merged.drop_duplicates(subset=["koi_name"], keep="first")

# Remove bad matches
merged = merged[merged["relative_diff"] < 0.05]

# Map labels to numbers
label_map = {
    "CONFIRMED": 1,
    "CANDIDATE": 1,
    "FALSE POSITIVE": 0,
    "NOT DISPOSITIONED": -1
}
merged["target"] = merged[label_col].map(label_map)
merged = merged[merged["target"] >= 0]

# Handle missing values
merged["odd_even_mismatch"] = merged["odd_even_mismatch"].fillna(0)
merged["SDE"] = merged["SDE"].fillna(0)

# Select final columns
final = merged[
    [
        "koi_name",
        "global_npy",
        "local_npy",
        "odd_even_mismatch",
        "SDE",
        "period_days",
        "target",
    ]
]

final.columns = [
    "koi_name",
    "global_path",
    "local_path",
    "odd_even_mismatch",
    "sde",
    "period",
    "label",
]

# Count classes
cont_confirmed = (final["label"] == 1).sum()
cont_fp = (final["label"] == 0).sum()

print(f"Confirmed/candidate planets: {cont_confirmed}")
print(f"False positives: {cont_fp}")

# Save output
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final.to_csv(output_path, index=False)
