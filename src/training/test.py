import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from dataloader import KeplerDataset
from cnn_model import KatnissNet
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, roc_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelos = [f for f in os.listdir('.') if f.endswith('.pth') and ('vloss' in f or '' in f)]

if not modelos:
    print("Nenhum modelo encontrado.")
    exit()

print("")
for i, m in enumerate(modelos):
    print(f"  {i+1}. {m}")

indice = int(input("\n ")) - 1
modelo_path = modelos[indice]
print(f"\ncarregando o modelo: {modelo_path}...\n")

model = KatnissNet(dropout_rate=0.3).to(device)
model.load_state_dict(torch.load(modelo_path, map_location=device))
model.eval()

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

test_path = os.path.join(project_root, "data", "processed", "test", "test_split.csv")
print("Test path:", test_path)

test_dataset = KeplerDataset(test_path, train_mode=False)
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

auc = roc_auc_score(all_labels, all_probs)
print(f"\nAUC-ROC Score: {auc:.4f}")

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
indice_otimo = np.argmax(tpr - fpr)
threshold_otimo = thresholds[indice_otimo]
print(f"thr Ideal Encontrado: {threshold_otimo:.4f}\n")

pred_otimizada = (all_probs > threshold_otimo).astype(int)
print("thr otimizado resultados:")
print(classification_report(all_labels, pred_otimizada, target_names=['Falso Positivo', 'Planeta']))

df_teste = test_dataset.data_frame.copy()
df_teste['predicao_modelo'] = pred_otimizada
impostores = df_teste[(df_teste['label'] == 0) & (df_teste['predicao_modelo'] == 1)]
impostores.to_csv("para_analise.csv", index=False)
