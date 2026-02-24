
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os


print("Random forest")

os.makedirs("./data/processed/classification_alg/models", exist_ok=True)

X_train = pd.read_csv("./data/processed/classification_alg/X_train.csv")
X_test = pd.read_csv("./data/processed/classification_alg/X_test.csv")
y_train = pd.read_csv("./data/processed/classification_alg/y_train.csv").squeeze()
y_test = pd.read_csv("./data/processed/classification_alg/y_test.csv").squeeze()

#Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)


best = grid.best_estimator_
# print("\nMelhores parâmetros:", grid.best_params_)
# print(f"ROC-AUC CV: {grid.best_score_:.4f}")

# Teste
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]

print("\nResultados no teste:")
print(classification_report(y_test, y_pred, target_names=['FP', 'Planeta']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")


importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best.feature_importances_
}).sort_values('importance', ascending=False)
# print("\n Top 5 features:")
# print(importances.head(5).to_string(index=False))


joblib.dump(best, "./data/processed/classification_alg/models/random_forest.pkl")
