from collections import Counter
import pandas as pd

def compute_class_weights(csv_path, label_column="label"):
    df = pd.read_csv(csv_path)
    labels = df[label_column].values
    label_counts = Counter(labels)

    total = sum(label_counts.values())
    n_classes = len(label_counts)

    weights = {
        label: total / (n_classes * count)
        for label, count in label_counts.items()
    }

    label_to_index = {label: idx for idx, label in enumerate(sorted(label_counts))}
    weights_indexed = {label_to_index[label]: weight for label, weight in weights.items()}

    return weights_indexed, label_to_index

# Exemplo de uso
weights, index_map = compute_class_weights("data/processed/dataset_final.csv", label_column="label")
print("Class Weights:", weights)
print("Label to Index:", index_map)
