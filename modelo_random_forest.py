import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

def main():
    DATASET = 'datasets/DATASETFINAL.csv'
    PASTA = 'graficos_modelo_rf'
    ALVO = 'OBITO'
    THRESHOLD = 0.6
    FEATURES_SELECIONADAS = [
        'SUPORT_VEN', 'IDADE_ANOS', 'UTI', 'CLASSI_FIN', 'FATOR_RISC',
        'VACINA_COV', 'HOSPITAL', 'SATURACAO', 'TOSSE', 'CARDIOPATI',
        'PCR_SARS2', 'CS_ESCOL_N'
    ]

    if os.path.exists(PASTA):
        shutil.rmtree(PASTA)
    os.makedirs(PASTA)

    df = pd.read_csv(DATASET, sep=';')

    colunas_ausentes = [c for c in FEATURES_SELECIONADAS + [ALVO] if c not in df.columns]
    if colunas_ausentes:
        raise ValueError(f'Colunas ausentes no dataset: {colunas_ausentes}')

    X = df[FEATURES_SELECIONADAS].apply(pd.to_numeric, errors='coerce').fillna(-1)
    y = pd.to_numeric(df[ALVO], errors='coerce')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    modelo = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    y_prob = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    relatorio = classification_report(y_test, y_pred, target_names=['Cura (0)', 'Óbito (1)'], zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    with open(f'{PASTA}/relatorio_rf.txt', 'w', encoding='utf-8') as f:
        f.write('Features usadas:\n')
        f.write('\n'.join(FEATURES_SELECIONADAS))
        f.write('\n\nDistribuição da variável alvo:\n')
        f.write(y.value_counts().sort_index().to_string())
        f.write(f'\n\nThreshold utilizado: {THRESHOLD}\n')
        f.write(f'\nROC AUC: {auc:.4f}\n\n')
        f.write(relatorio)

    # Matriz de confusão
    matriz = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=['Cura', 'Óbito'], yticklabels=['Cura', 'Óbito'])
    plt.title('Matriz de Confusão - Random Forest')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(f'{PASTA}/matriz_confusao_rf.png', dpi=150)
    plt.close()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Aleatório')
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Curva ROC - Random Forest')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{PASTA}/curva_roc_rf.png', dpi=150)
    plt.close()
    
    # Importância das features
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    features_ord = [FEATURES_SELECIONADAS[i] for i in indices]
    valores_ord = importancias[indices]

    plt.figure(figsize=(10, 10))
    barras = plt.barh(features_ord[::-1], valores_ord[::-1], color='steelblue')
    plt.xlabel('Importância')
    plt.title('Importância das Features - Random Forest')
    for barra, valor in zip(barras, valores_ord[::-1]):
        plt.text(barra.get_width() + 0.002, barra.get_y() + barra.get_height() / 2, f'{valor:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{PASTA}/feature_importance_rf.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    main()