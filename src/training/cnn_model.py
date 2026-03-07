# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ExoplanetModel(nn.Module):
#     def __init__(self):
#         super(ExoplanetModel, self).__init__()
        
#         self.global_branch = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(16, 32, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Flatten()
#         )
        
#         self.local_branch = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Flatten()
#         )

#         self.fc1 = nn.Linear(32 * 500 + 32 * 250 + 3, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.output = nn.Linear(32, 1)

#     def forward(self, x_global, x_local, x_aux):
#         g = self.global_branch(x_global)
#         l = self.local_branch(x_local)
#         combined = torch.cat((g, l, x_aux), dim=1)
#         x = F.relu(self.fc1(combined))
#         x = F.relu(self.fc2(x))
#         return self.output(x)

import torch
import torch.nn as nn
import torch.nn.functional as F


class KatnissNet(nn.Module):
    def __init__(self, dropout_rate=0.55):
        super(KatnissNet, self).__init__()
        self.global_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveMaxPool1d(16),
            nn.Flatten()
        )

        self.local_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveMaxPool1d(16),
            nn.Flatten()
        )

    
        self.fc_aux = nn.Sequential(
            nn.BatchNorm1d(14),
            nn.Linear(14, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.fc1 = nn.Linear(2080, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(32, 1)

    def forward(self, x_global, x_local, x_aux):

        g = self.global_branch(x_global)
        l = self.local_branch(x_local)
        x_aux_norm = self.fc_aux(x_aux)

        combined = torch.cat((g, l, x_aux_norm), dim=1)

        x = F.relu(self.bn1(self.fc1(combined)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        return self.output(x)