import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv('datasets/DATASETFINAL.csv', sep=';')

features = [
    'FAIXA_ETARIA', 'CS_SEXO',
    'PUERPERA', 'CARDIOPATI', 'HEMATOLOGI', 'SIND_DOWN',
    'HEPATICA', 'ASMA', 'DIABETES', 'NEUROLOGIC', 'PNEUMOPATI',
    'IMUNODEPRE', 'RENAL', 'OBESIDADE',
    'DISPNEIA', 'SATURACAO', 'DESC_RESP',
    'UTI', 'SUPORT_VEN'
]

X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['OBITO']

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_real, y_train_real)
y_pred1 = rf.predict(X_test_real)
cm1 = confusion_matrix(y_test_real, y_pred1)

print(classification_report(y_test_real, y_pred1, target_names=['Cura (0)', 'Óbito (1)']))

pasta = "graficos_desbalanceado"
os.makedirs(pasta, exist_ok=True)

# GRÁFICO 1 — Matriz de Confusão
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm1,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Sobreviveu', 'Óbito'],
    yticklabels=['Sobreviveu', 'Óbito']
)
plt.title('Matriz de Confusão - Random Forest')
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.tight_layout()
plt.savefig(f'{pasta}/matriz_confusao_rf.png', dpi=150)

# GRÁFICO 2 — Curva ROC
y_prob = rf.predict_proba(X_test_real)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_real, y_prob)
auc = roc_auc_score(y_test_real, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Aleatório')
plt.xlabel('Taxa de Falso Positivo (FPR)')
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
plt.title('Curva ROC - Random Forest')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{pasta}/curva_roc_rf.png', dpi=150)

# GRÁFICO 3 — Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features_ordenadas = [features[i] for i in indices]
valores_ordenados  = importances[indices]

plt.figure(figsize=(10, 6))
bars = plt.barh(features_ordenadas[::-1], valores_ordenados[::-1], color='steelblue')
plt.xlabel('Importância')
plt.title('Features que mais impactaram no Óbito')

for bar, val in zip(bars, valores_ordenados[::-1]):
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
             f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{pasta}/feature_importance_rf.png', dpi=150)