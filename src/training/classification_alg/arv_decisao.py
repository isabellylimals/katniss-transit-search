
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os


print("Arvore de Decisão")

os.makedirs("./data/processed/classification_alg/models", exist_ok=True)

X_train = pd.read_csv("./data/processed/classification_alg/X_train.csv")
X_test = pd.read_csv("./data/processed/classification_alg/X_test.csv")
y_train = pd.read_csv("./data/processed/classification_alg/y_train.csv").squeeze()
y_test = pd.read_csv("./data/processed/classification_alg/y_test.csv").squeeze()

#Grid Search
param_grid = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'class_weight': ['balanced', None]
}

dt = DecisionTreeClassifier(random_state=42)
grid = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)


best = grid.best_estimator_
# print("\nMelhores parâmetros:", grid.best_params_)
# print(f"ROC-AUC CV: {grid.best_score_:.4f}")

#Testee
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]

print("\nResultados no teste:")
print(classification_report(y_test, y_pred, target_names=['FP', 'Planeta']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

#Salvar
joblib.dump(best, "./data/processed/classification_alg/models/arvore_decisao.pkl")
