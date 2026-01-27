import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class KeplerDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]

        global_path = row["global_path"]
        if not os.path.exists(global_path):
            global_path = os.path.join(".", global_path)

        try:
            global_view = np.load(global_path).astype(np.float32)
            if np.isnan(global_view).any():
                global_view = np.nan_to_num(global_view)
        except Exception:
            print(f"Warning: failed to load global view {global_path}, using zeros")
            global_view = np.zeros(2001, dtype=np.float32)

        local_path = row["local_path"]
        if not os.path.exists(local_path):
            local_path = os.path.join(".", local_path)

        try:
            local_view = np.load(local_path).astype(np.float32)
            if np.isnan(local_view).any():
                local_view = np.nan_to_num(local_view)
        except Exception:
            print(f"Warning: failed to load local view {local_path}, using zeros")
            local_view = np.zeros(1001, dtype=np.float32)

        aux_features = np.array(
            [
                row["sde"],
                row["period"] / 100.0,
                row["odd_even_mismatch"],
            ],
            dtype=np.float32,
        )

        label = np.array([row["label"]], dtype=np.float32)

        return {
            "global": torch.tensor(global_view).unsqueeze(0),
            "local": torch.tensor(local_view).unsqueeze(0),
            "aux": torch.tensor(aux_features),
            "label": torch.tensor(label),
        }
