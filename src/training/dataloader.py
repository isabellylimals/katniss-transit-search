import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d

class KeplerDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_frame)

    def resample_signal(self, signal, target_size):
        if len(signal) == target_size:
            return signal
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, target_size)
        f = interp1d(x_old, signal, kind='linear', fill_value="extrapolate")
        return f(x_new).astype(np.float32)

    def normalize(self, signal):
        std = np.std(signal)
        if std > 0:
            return (signal - np.mean(signal)) / std
        return signal - np.mean(signal)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        
        try:
            g_view = np.load(row["global_path"]).astype(np.float32)
            g_view = np.nan_to_num(g_view, nan=0.0, posinf=0.0, neginf=0.0)
            g_view = self.resample_signal(g_view, 2001)
            g_view = self.normalize(g_view)
        except:
            g_view = np.zeros(2001, dtype=np.float32)

        try:
            l_view = np.load(row["local_path"]).astype(np.float32)
            l_view = np.nan_to_num(l_view, nan=0.0, posinf=0.0, neginf=0.0)
            l_view = self.resample_signal(l_view, 1001)
            l_view = self.normalize(l_view)
        except:
            l_view = np.zeros(1001, dtype=np.float32)

        aux = np.array([row["sde"], row["period"] / 100.0, row["odd_even_mismatch"]], dtype=np.float32)
        label = np.array([row["label"]], dtype=np.float32)

        return {
            "global": torch.tensor(g_view).unsqueeze(0),
            "local": torch.tensor(l_view).unsqueeze(0),
            "aux": torch.tensor(aux),
            "label": torch.tensor(label),
        }