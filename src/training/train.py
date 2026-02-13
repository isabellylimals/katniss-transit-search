import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import KeplerDataset
from cnn_model import ExoplanetModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = KeplerDataset("data/processed/train/train_split.csv")
    test_dataset = KeplerDataset("data/processed/test/test_split.csv")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = ExoplanetModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(20):
        model.train()
        r_loss = 0.0
        for batch in train_loader:
            g, l, a, labels = batch["global"].to(device), batch["local"].to(device), \
                             batch["aux"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(g, l, a)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            r_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for b in test_loader:
                outputs = model(b["global"].to(device), b["local"].to(device), b["aux"].to(device))
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                labels = b["label"].to(device)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1} | Loss: {r_loss/len(train_loader):.4f} | Acc: {100*correct/total:.2f}%")
    
    model_name = f"exoplanet_model_epoch{epoch+1}_acc{correct/total:.2f}.pth"

    torch.save(model.state_dict(), model_name)
    print(f"Modelo salvo como {model_name}")

if __name__ == "__main__":
    train()