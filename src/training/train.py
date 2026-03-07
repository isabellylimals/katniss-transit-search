# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from dataloader import KeplerDataset
# from cnn_model import ExoplanetModel

# def train():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     train_dataset = KeplerDataset("data/processed/train/train_split.csv")
#     test_dataset = KeplerDataset("data/processed/test/test_split.csv")
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#     model = ExoplanetModel().to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.0001)

#     for epoch in range(20):
#         model.train()
#         r_loss = 0.0
#         for batch in train_loader:
#             g, l, a, labels = batch["global"].to(device), batch["local"].to(device), \
#                              batch["aux"].to(device), batch["label"].to(device)

#             optimizer.zero_grad()
#             outputs = model(g, l, a)
#             loss = criterion(outputs, labels)
            
#             if torch.isnan(loss):
#                 continue
                
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             r_loss += loss.item()

#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for b in test_loader:
#                 outputs = model(b["global"].to(device), b["local"].to(device), b["aux"].to(device))
#                 predicted = (torch.sigmoid(outputs) > 0.5).float()
#                 labels = b["label"].to(device)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         print(f"Epoch {epoch+1} | Loss: {r_loss/len(train_loader):.4f} | Acc: {100*correct/total:.2f}%")
    
#     model_name = f"exoplanet_model_epoch{epoch+1}_acc{correct/total:.2f}.pth"

#     torch.save(model.state_dict(), model_name)
#     print(f"Modelo salvo como {model_name}")

# if __name__ == "__main__":
#     train()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from cnn_model import KatnissNet 
from dataloader import KeplerDataset
import numpy as np
import os
print("testando treino script 123 123 123")

class BalancedLabelSmoothingLoss(nn.Module):
    def __init__(self, pos_weight=None, smoothing=0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.smoothing = smoothing
        
    def forward(self, pred, target):
    
        smooth_target = target * (1 - self.smoothing) + self.smoothing * 0.5
        
        if self.pos_weight is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                pred, smooth_target, pos_weight=self.pos_weight
            )
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(
                pred, smooth_target
            )
        return loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = KeplerDataset("data/processed/train/train_split.csv", train_mode=True)
    test_dataset = KeplerDataset("data/processed/test/test_split.csv", train_mode=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)


    num_planetas = sum(train_dataset.data_frame['label'] == 1)
    num_falsos = sum(train_dataset.data_frame['label'] == 0)

    peso_planeta = num_falsos / (num_planetas + 1e-5) 
    
    pos_weight = torch.tensor([peso_planeta]).to(device)
    model = KatnissNet(dropout_rate=0.3).to(device)
    
    criterion = BalancedLabelSmoothingLoss(pos_weight=pos_weight, smoothing=0.1) 
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=3e-4)  
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4) #o certo
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(30):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            g = batch["global"].to(device)
            l = batch["local"].to(device)
            a = batch["aux"].to(device)
            y = batch["label"].to(device).view(-1, 1)
            
            optimizer.zero_grad()
            logits = model(g, l, a)
            loss = criterion(logits, y)
        
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
    
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                g = batch["global"].to(device)
                l = batch["local"].to(device)
                a = batch["aux"].to(device)
                y = batch["label"].to(device).view(-1, 1)
                
                logits = model(g, l, a)
                v_loss = criterion(logits, y)
                val_losses.append(v_loss.item())
                
                pred = (torch.sigmoid(logits) > 0.50).float()
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        
        acc = correct / total
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        avg_val_loss = np.mean(val_losses) if val_losses else 0
        
        print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {100*acc:.2f}%")
        
    
        scheduler.step(avg_val_loss)
    
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"melhor_modelo_vloss{avg_val_loss:.4f}.pth")
    
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n Parando na época {epoch+1}")
                break
    model.load_state_dict(torch.load(f"melhor_modelo_vloss{best_val_loss:.4f}.pth"))
    model.eval()
    
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            g = batch["global"].to(device)
            l = batch["local"].to(device)
            a = batch["aux"].to(device)
            y = batch["label"].to(device).view(-1, 1)
            
            # Predição normal (Baseline)
            logits = model(g, l, a)
            pred_normal = (torch.sigmoid(logits) > 0.50).float()

            total += y.size(0)
  

    return best_val_loss

if __name__ == "__main__":
    train()