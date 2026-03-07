# import torch
# from torch.utils.data import Dataset
# import pandas as pd
# import numpy as np
# import os
# from scipy.interpolate import interp1d

# class KeplerDataset(Dataset):
#     def __init__(self, csv_file):
#         self.data_frame = pd.read_csv(csv_file)

#     def __len__(self):
#         return len(self.data_frame)

#     def resample_signal(self, signal, target_size):
#         if len(signal) == target_size:
#             return signal
#         x_old = np.linspace(0, 1, len(signal))
#         x_new = np.linspace(0, 1, target_size)
#         f = interp1d(x_old, signal, kind='linear', fill_value="extrapolate")
#         return f(x_new).astype(np.float32)

#     def normalize(self, signal):
#         std = np.std(signal)
#         if std > 0:
#             return (signal - np.mean(signal)) / std
#         return signal - np.mean(signal)

#     def __getitem__(self, idx):
#         row = self.data_frame.iloc[idx]
        
#         try:
#             g_view = np.load(row["global_path"]).astype(np.float32)
#             g_view = np.nan_to_num(g_view, nan=0.0, posinf=0.0, neginf=0.0)
#             g_view = self.resample_signal(g_view, 2001)
#             g_view = self.normalize(g_view)
#         except:
#             g_view = np.zeros(2001, dtype=np.float32)

#         try:
#             l_view = np.load(row["local_path"]).astype(np.float32)
#             l_view = np.nan_to_num(l_view, nan=0.0, posinf=0.0, neginf=0.0)
#             l_view = self.resample_signal(l_view, 1001)
#             l_view = self.normalize(l_view)
#         except:
#             l_view = np.zeros(1001, dtype=np.float32)

#         aux = np.array([row["sde"], row["period"] / 100.0, row["odd_even_mismatch"]], dtype=np.float32)
#         label = np.array([row["label"]], dtype=np.float32)

#         return {
#             "global": torch.tensor(g_view).unsqueeze(0),
#             "local": torch.tensor(l_view).unsqueeze(0),
#             "aux": torch.tensor(aux),
#             "label": torch.tensor(label),
#         }

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
print("começo")

class KeplerDataset(Dataset):
    def __init__(self, csv_file, train_mode=True):
        self.data_frame = pd.read_csv(csv_file)
        self.train_mode = train_mode
        
        # Calcular mediana dos falsos positivos para substituir infs
        mask_falsos = self.data_frame['label'] == 0
        mismatches_falsos = self.data_frame.loc[mask_falsos, 'odd_even_mismatch']
        # Substituir inf temporariamente para calcular mediana
        mismatches_clean = mismatches_falsos.replace([np.inf, -np.inf], np.nan)
        self.mediana_mismatch_falsos = mismatches_clean.median()
        
        if np.isnan(self.mediana_mismatch_falsos):
            self.mediana_mismatch_falsos = 0.7  
        
        if self.train_mode:
            print(f"Mediana do mismatch (Treino): {self.mediana_mismatch_falsos:.4f}")

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
    
    def tratar_mismatch(self, mismatch_val, label):
    
        if np.isinf(mismatch_val) or np.isnan(mismatch_val):
            #
            if label == 0:
                return self.mediana_mismatch_falsos
            else:
                
                return 5.0
        
   
        if mismatch_val > 100:
           
            return 100.0

        if mismatch_val < 0:
            return abs(mismatch_val)  
        
        return mismatch_val
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

        if self.train_mode:
            
            if np.random.rand() > 0.5:
                g_view += np.random.normal(0, 0.005, g_view.shape).astype(np.float32)
                l_view += np.random.normal(0, 0.005, l_view.shape).astype(np.float32)
            
           
            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.9, 1.1)
                g_view = (g_view * scale).astype(np.float32)
                l_view = (l_view * scale).astype(np.float32)
                
            
            if np.random.rand() > 0.5:
                shift = np.random.randint(-3, 4)
                g_view = np.roll(g_view, shift)
                l_view = np.roll(l_view, shift)

        
        sde_val = np.nan_to_num(row["sde"], nan=0.0, posinf=0.0, neginf=0.0)
        period_val = np.nan_to_num(row["period"], nan=0.0, posinf=0.0, neginf=0.0)
        mismatch_val = np.nan_to_num(row["odd_even_mismatch"], nan=0.0, posinf=0.0, neginf=0.0)

        
        if np.isinf(mismatch_val) or np.isnan(mismatch_val):
            mismatch_val = self.mediana_mismatch_falsos
        mismatch_val = np.clip(mismatch_val, 0, 100)

        eps = 1e-8
        
        
        log_period = np.log1p(np.clip(period_val, 0, 1000))  
        log_sde = np.log1p(np.clip(sde_val, 0, 1000))       
        
        
        sde_div_period = np.clip(sde_val / (period_val + eps), -10, 10)
        period_div_sde = np.clip(period_val / (sde_val + eps), -10, 10)
        odd_div_sde = np.clip(mismatch_val / (sde_val + eps), -10, 10)
        sde_div_mismatch = sde_val / (mismatch_val + eps)
        
        sde_x_logperiod = sde_val * log_period
        sde_x_odd = sde_val * mismatch_val
        logperiod_x_odd = log_period * mismatch_val
        
        
        period_curva = np.sin(2 * np.pi * period_val / 365) if period_val > 0 else 0
        period_curva2 = np.cos(2 * np.pi * period_val / 365) if period_val > 0 else 0
    
        sde_squared = (sde_val ** 2) / 1000
        
        aux = np.array([
            sde_val,              
            log_period,           
            mismatch_val,         
            sde_x_logperiod,      
            sde_div_period,       
            period_div_sde,       
            log_sde,             
            sde_x_odd,            
            logperiod_x_odd,      
            sde_squared,          
            odd_div_sde,          
            period_curva,         
            period_curva2,
            sde_div_mismatch       
        ], dtype=np.float32)
        
        
        aux = np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0)
        
        
        label = np.array([label_val], dtype=np.float32)

        return {
            "global": torch.tensor(g_view).unsqueeze(0),
            "local": torch.tensor(l_view).unsqueeze(0),
            "aux": torch.tensor(aux),
            "label": torch.tensor(label)
        }
        