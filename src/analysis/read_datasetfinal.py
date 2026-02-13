import pandas as pd

def analyze_dataset(csv_path, label_column="label"):
    df = pd.read_csv(csv_path)
    
    counts = df[label_column].value_counts().sort_index()
    total = len(df)
    
    print("-" * 30)
    print(f"Total de amostras: {total}")
    print("-" * 30)
    
    for label, count in counts.items():
        percentage = (count / total) * 100
        status = "Exoplanetas (PC/CONF)" if label == 1 else "Falsos Positivos/Ruído"
        print(f"Classe {label} ({status}):")
        print(f"  -> Quantidade: {count}")
        print(f"  -> Proporção:  {percentage:.2f}%")
        print("-" * 30)

analyze_dataset("data/processed/dataset_final.csv")