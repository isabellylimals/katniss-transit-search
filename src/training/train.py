import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from cnn_model import KatnissNet 
from dataloader import KeplerDataset
import numpy as np
import os


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing
    def forward(self, inputs, targets):
        smooth_target = targets * (1 - self.smoothing) + self.smoothing * 0.5
        bce_loss = F.binary_cross_entropy_with_logits(inputs, smooth_target, reduction='none')
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt)**self.gamma * bce_loss).mean()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    train_path = os.path.join(project_root, "data", "processed", "train", "train_split.csv")
    test_path = os.path.join(project_root, "data", "processed", "test", "test_split.csv")

    train_dataset = KeplerDataset(train_path, train_mode=True)
    test_dataset = KeplerDataset(test_path, train_mode=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = KatnissNet(dropout_rate=0.3).to(device)
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    patience, patience_counter = 12, 0
    

    top_3_modelos = []

    for epoch in range(50):
        model.train()
        train_losses = []
        for batch in train_loader:
            g, l, a, y = batch["global"].to(device), batch["local"].to(device), batch["aux"].to(device), batch["label"].to(device).view(-1, 1)
            
            if np.random.rand() < 0.2:
                a = torch.zeros_like(a)
    
            optimizer.zero_grad()
            logits = model(g, l, a)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        
        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for batch in test_loader:
                g, l, a, y = batch["global"].to(device), batch["local"].to(device), batch["aux"].to(device), batch["label"].to(device).view(-1, 1)
                logits = model(g, l, a)
                val_losses.append(criterion(logits, y).item())
                pred = (torch.sigmoid(logits) > 0.5).float()
                correct += (pred == y).sum().item()
                total += y.size(0)

        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1:2d} | Val Loss: {avg_val_loss:.4f} | Acc: {100*(correct/total):.2f}%")
        scheduler.step(avg_val_loss)

        if len(top_3_modelos) < 3 or avg_val_loss < top_3_modelos[-1][0]:
            nome_arquivo = f"modelo_top_ep{epoch+1}_vloss{avg_val_loss:.4f}.pth"
            torch.save(model.state_dict(), nome_arquivo)
        
            top_3_modelos.append((avg_val_loss, nome_arquivo))
            top_3_modelos.sort(key=lambda x: x[0])
            
        
            if len(top_3_modelos) > 3:
                pior_loss, pior_arquivo = top_3_modelos.pop()
                if os.path.exists(pior_arquivo):
                    os.remove(pior_arquivo)

    
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "melhor_modelo_FINAL.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience: 
                print(f"\n{epoch+1} Early Stopping.")
                break


    for i, (loss, nome) in enumerate(top_3_modelos):
        print(f"{i+1}º Lugar: {nome} (Loss: {loss:.4f})")

    if os.path.exists("melhor_modelo_FINAL.pth"):
        model.load_state_dict(torch.load("melhor_modelo_FINAL.pth"))
    
    return best_val_loss

if __name__ == "__main__":
    train()