
import torch
import torch.nn as nn
import torch.nn.functional as F

class KatnissNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(KatnissNet, self).__init__()
        
  
        self.global_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, stride=2), 
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 64, kernel_size=5, padding=2), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.AdaptiveMaxPool1d(16), 
            nn.Flatten()
        )

        
        self.local_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.AdaptiveMaxPool1d(16), 
            nn.Flatten()
        )

    
        self.fc_aux = nn.Sequential(
            nn.BatchNorm1d(8),
            nn.Linear(8, 256), 
            nn.ReLU(),
            nn.Dropout(0.6)
        )

        self.cnn_compress = nn.Sequential(
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # self.fc_final = nn.Sequential(
        #     nn.Linear(192, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate)
        # )


        self.fc_final = nn.Sequential(
            nn.Linear(384, 128),     
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.output = nn.Linear(128, 1)
    def forward(self, x_global, x_local, x_aux):

        g = self.global_branch(x_global)
        l = self.local_branch(x_local)
        
        cnn_combined = torch.cat((g, l), dim=1) 
        cnn_features = self.cnn_compress(cnn_combined)
        
        aux_features = self.fc_aux(x_aux) # Fica tamanho 64
        
        final_combined = torch.cat((cnn_features, aux_features), dim=1) 
        
   
        x = self.fc_final(final_combined)
        return self.output(x)