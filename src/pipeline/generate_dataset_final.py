import os
import pandas as pd

metadata_path = "./data/processed/metadata.csv"
koi_template_path = "./data/raw/koi_template.csv"
output_path = "./data/processed/dataset_final.csv"

# Carrega os arquivos
meta = pd.read_csv(metadata_path)
koi = pd.read_csv(koi_template_path, comment="#", engine="python")
koi.columns = [c.lower().strip() for c in koi.columns]

# Identifica colunas de ID e rótulo
id_col = next((c for c in koi.columns if "kepid" in c), "kepid")
label_col = next((c for c in koi.columns if "label" in c or "disposition" in c), "koi_disposition")

# Mescla usando kic_id para trazer os rótulos
merged = pd.merge(
    meta,
    koi[[id_col, label_col]].rename(columns={id_col: "kic_id"}),
    on="kic_id",
    how="inner"
)

# Seleciona colunas para o dataset final
final = merged[["koi_name", "global_npy", "local_npy", label_col]]
final.columns = ["koi_name", "global_path", "local_path", "label"]
final = final.drop_duplicates()
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final.to_csv(output_path, index=False)


