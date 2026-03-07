
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from dataloader import KeplerDataset
from cnn_model import KatnissNet 
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelos = [f for f in os.listdir('.') if f.endswith('.pth') and 'vloss' in f]

if not modelos:
    print("Nenhum modelo encontrado.")
    exit()

for i, m in enumerate(modelos):
    print(f"  {i+1}. {m}")


indice = int(input("\n")) - 1
modelo_path = modelos[indice]

model = KatnissNet(dropout_rate=0.3).to(device)

model.load_state_dict(torch.load(modelo_path, map_location=device))
model.eval()

test_dataset = KeplerDataset("data/processed/test/test_split.csv", train_mode=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_probs = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        g = batch["global"].to(device)
        l = batch["local"].to(device)
        a = batch["aux"].to(device)
        
        outputs = model(g, l, a)
        probs = torch.sigmoid(outputs).cpu().numpy()
        
        all_probs.extend(probs.flatten())
        all_labels.extend(batch["label"].numpy().flatten())


all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

thresh_padrao = 0.50
pred_padrao = (all_probs > thresh_padrao).astype(int)

print(classification_report(all_labels, pred_padrao, target_names=['Falso Positivo', 'Planeta']))

auc = roc_auc_score(all_labels, all_probs)
print(f"AUC-ROC Score: {auc:.4f}")
