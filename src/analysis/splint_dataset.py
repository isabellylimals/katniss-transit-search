import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/dataset_final.csv")

train_df, test_df = train_test_split(
    df, 
    test_size=0.20, 
    random_state=42, 
    stratify=df['label']
)

train_df.to_csv("data/processed/train/train_split.csv", index=False)
test_df.to_csv("data/processed/test/test_split.csv", index=False)

print("Arquivos train_split.csv e test_split.csv gerados!")