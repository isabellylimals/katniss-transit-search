import torch
import torch.nn as nn
import torch.nn.functional as F

class ExoplanetModel(nn.Module):
    def __init__(self):
        super(ExoplanetModel, self).__init__()
        
        self.global_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        self.local_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(32 * 500 + 32 * 250 + 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x_global, x_local, x_aux):
        g = self.global_branch(x_global)
        l = self.local_branch(x_local)
        combined = torch.cat((g, l, x_aux), dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.output(x)