import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE

TIPO = input("Digite o tipo de modelo (1-desbalanceado, 2-undersampling, 3-oversampling): ")

if TIPO not in ['1', '2', '3']:
    raise ValueError("Tipo inválido. Digite 1, 2 ou 3.")

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

if TIPO == '1':
    X_train_bal = X_train_real
    y_train_bal = y_train_real

elif TIPO == '2':
    train_data = X_train_real.copy()
    train_data['OBITO'] = y_train_real
    
    df_obito = train_data[train_data['OBITO'] == 1]
    df_cura  = train_data[train_data['OBITO'] == 0].sample(n=len(df_obito), random_state=42)
    
    train_balanceado = pd.concat([df_obito, df_cura]).sample(frac=1, random_state=42)
    
    X_train_bal = train_balanceado.drop('OBITO', axis=1)
    y_train_bal = train_balanceado['OBITO']

elif TIPO == '3':
    n_curas_treino  = (y_train_real == 0).sum()
    n_obitos_alvo   = int(n_curas_treino * (40 / 60))
    smote = SMOTE(sampling_strategy={1: n_obitos_alvo}, random_state=42, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_real, y_train_real)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_bal, y_train_bal)

y_pred = rf.predict(X_test_real)
cm1 = confusion_matrix(y_test_real, y_pred)

print(classification_report(y_test_real, y_pred, target_names=['Cura (0)', 'Óbito (1)']))

pasta = "graficos_desbalanceado"
if TIPO == '2':
    pasta = "graficos_undersampling"
elif TIPO == '3':
    pasta = "graficos_oversampling"

os.makedirs(pasta, exist_ok=True)

color = 'Blues'
color_2 = 'steelblue'

if TIPO == '2':
    color = 'Greens'
    color_2 = 'seagreen'
elif TIPO == '3':
    color = 'Oranges'
    color_2 = 'darkorange'

# GRÁFICO 1 — Matriz de Confusão
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm1,
    annot=True,
    fmt='d',
    cmap=color,
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
plt.plot(fpr, tpr, color=color_2, lw=2, label=f'AUC = {auc:.3f}')
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
bars = plt.barh(features_ordenadas[::-1], valores_ordenados[::-1], color=color_2)
plt.xlabel('Importância')
plt.title('Features que mais impactaram no Óbito')

for bar, val in zip(bars, valores_ordenados[::-1]):
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
             f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{pasta}/feature_importance_rf.png', dpi=150)