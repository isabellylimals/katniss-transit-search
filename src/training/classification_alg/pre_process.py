
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
#OBS:Os dados do dataset_rf.csv( o que eu usei apenas para os algortimos de classificação) 
# eu coloquei no drive na pasta processed
df = pd.read_csv("./data/processed/dataset_rf.csv")
print(f"Dataset carregado: {df.shape}")

# Features
feature_cols = [col for col in df.columns if col not in ['label', 'koi_name', 'kic_id']]
X = df[feature_cols].copy()
y = df['label'].copy()

print(f"\nFeatures: {len(feature_cols)}")
print(f"Distribuição: {y.value_counts().to_dict()}")


print("\npré-processamento")


X = X.replace([np.inf, -np.inf], np.nan)

for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())


for col in X.columns:
    med = X[col].median()
    std = X[col].std()
    if std > 0:
        X[col] = X[col].clip(med - 3*std, med + 3*std)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nSplit: Treino {X_train.shape}, Teste {X_test.shape}")

X_train.to_csv("./data/processed/classification_alg/X_train.csv", index=False)
X_test.to_csv("./data/processed/classification_alg/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("./data/processed/classification_alg/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("./data/processed/classification_alg/y_test.csv", index=False)

print("\nDados salvos em ./data/processed/classification_alg/")