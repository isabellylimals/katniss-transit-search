import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

print("")

class KeplerDataset(Dataset):
    def __init__(self, csv_file, train_mode=True):
        self.data_frame = pd.read_csv(csv_file)
        self.train_mode = train_mode
      
        mask_falsos = self.data_frame['label'] == 0
        mismatches_falsos = self.data_frame.loc[mask_falsos, 'odd_even_mismatch']
        mismatches_clean = mismatches_falsos.replace([np.inf, -np.inf], np.nan)
        self.mediana_mismatch_falsos = mismatches_clean.median()
        if np.isnan(self.mediana_mismatch_falsos):
            self.mediana_mismatch_falsos = 0.7  
            
       
        sde_valido = self.data_frame.loc[self.data_frame['sde'] > 0, 'sde']
        self.mediana_sde = sde_valido.median() if not sde_valido.empty else 8.5
        
        if self.train_mode:
            print(f"Mediana Mismatch (Falsos): {self.mediana_mismatch_falsos:.4f}")
            print(f"Mediana SDE (Camuflagem): {self.mediana_sde:.4f}")

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
        label_val = float(row["label"])


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


        #substituir sde pela mediana 
        sde_val = float(row["sde"])
        if sde_val == 0.0 or np.isnan(sde_val) or np.isinf(sde_val):
            sde_val = self.mediana_sde
            
        period_val = np.nan_to_num(row["period"], nan=0.0)
        mismatch_val = np.nan_to_num(row["odd_even_mismatch"], nan=0.0, posinf=self.mediana_mismatch_falsos, neginf=0.0)
        mismatch_val = np.clip(mismatch_val, 0, 100)

        snr_val = np.nan_to_num(row["koi_model_snr"], nan=0.0)
        impact_val = np.nan_to_num(row["koi_impact"], nan=0.0)
        depth_val = np.nan_to_num(row["koi_depth"], nan=0.0)
        prad_val = np.nan_to_num(row["koi_prad"], nan=0.0)
        
        log_period = np.log1p(np.clip(period_val, 0, 1000))
        log_sde = np.log1p(np.clip(sde_val, 0, 1000))

       
        aux = np.array([
            sde_val,              
            log_period,           
            mismatch_val,         
            log_sde,              
            snr_val,             
            impact_val,
            depth_val,            
            prad_val        
        ], dtype=np.float32)
        
        aux = np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0)
        

        if self.train_mode and torch.rand(1).item() < 0.25:
            aux = np.zeros_like(aux)

        label = np.array([label_val], dtype=np.float32)

        return {
            "global": torch.tensor(g_view).unsqueeze(0),
            "local": torch.tensor(l_view).unsqueeze(0),
            "aux": torch.tensor(aux),
            "label": torch.tensor(label)
        }
        
