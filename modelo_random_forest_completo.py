import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

DATASET = 'datasets/DATASETFINAL_COMPLETO.csv'
PASTA_SAIDA = 'graficos_modelo_rf_completo'
ALVO = 'OBITO'
FEATURES_SELECIONADAS = [
    'SUPORT_VEN', 'IDADE_ANOS', 'UTI', 'CLASSI_FIN', 'FATOR_RISC',
    'VACINA_COV', 'HOSPITAL', 'SATURACAO', 'TOSSE', 'CARDIOPATI',
    'PCR_SARS2', 'CS_ESCOL_N'
]


def avaliar_thresholds(y_real, y_prob):
    linhas = []

    for threshold in np.arange(0.25, 0.76, 0.05):
        y_pred = (y_prob >= threshold).astype(int)
        relatorio = classification_report(y_real, y_pred, output_dict=True, zero_division=0)
        obito = relatorio['1']

        linhas.append({
            'threshold': round(float(threshold), 2),
            'precision_obito': obito['precision'],
            'recall_obito': obito['recall'],
            'f1_obito': obito['f1-score'],
            'accuracy': relatorio['accuracy'],
        })

    return pd.DataFrame(linhas)


def salvar_matriz_confusao(y_real, y_pred):
    matriz = confusion_matrix(y_real, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matriz,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Cura', 'Óbito'],
        yticklabels=['Cura', 'Óbito'],
    )
    plt.title('Matriz de Confusão - Random Forest Completo')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/matriz_confusao_rf_completo.png', dpi=150)
    plt.close()


def salvar_curva_roc(y_real, y_prob):
    fpr, tpr, _ = roc_curve(y_real, y_prob)
    auc = roc_auc_score(y_real, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Aleatório')
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Curva ROC - Random Forest Completo')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/curva_roc_rf_completo.png', dpi=150)
    plt.close()


def salvar_curva_precision_recall(y_real, y_prob):
    precision, recall, _ = precision_recall_curve(y_real, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color='steelblue', lw=2)
    plt.xlabel('Recall - Óbito')
    plt.ylabel('Precision - Óbito')
    plt.title('Curva Precision-Recall - Random Forest Completo')
    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/curva_precision_recall_rf_completo.png', dpi=150)
    plt.close()


def salvar_importancia_features(modelo, features):
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    features_ordenadas = [features[i] for i in indices]
    valores_ordenados = importancias[indices]

    plt.figure(figsize=(10, 10))
    barras = plt.barh(features_ordenadas[::-1], valores_ordenados[::-1], color='steelblue')
    plt.xlabel('Importância')
    plt.title('Importância das Features - Random Forest Completo')

    for barra, valor in zip(barras, valores_ordenados[::-1]):
        plt.text(
            barra.get_width() + 0.002,
            barra.get_y() + barra.get_height() / 2,
            f'{valor:.3f}',
            va='center',
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/feature_importance_rf_completo.png', dpi=150)
    plt.close()


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    df = pd.read_csv(DATASET, sep=';')

    colunas_ausentes = [coluna for coluna in FEATURES_SELECIONADAS + [ALVO] if coluna not in df.columns]
    if colunas_ausentes:
        raise ValueError(f'Colunas ausentes no dataset: {colunas_ausentes}')

    features = FEATURES_SELECIONADAS
    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(-1)
    y = pd.to_numeric(df[ALVO], errors='coerce')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

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
    tabela_thresholds = avaliar_thresholds(y_test, y_prob)
    melhor_linha = tabela_thresholds.sort_values('f1_obito', ascending=False).iloc[0]
    melhor_threshold = melhor_linha['threshold']
    y_pred = (y_prob >= melhor_threshold).astype(int)

    relatorio = classification_report(
        y_test,
        y_pred,
        target_names=['Cura (0)', 'Óbito (1)'],
        zero_division=0,
    )
    auc = roc_auc_score(y_test, y_prob)

    print('Features usadas:')
    print(features)
    print('\nDistribuição da variável alvo:')
    print(y.value_counts().sort_index())
    print('\nMelhor threshold por F1 da classe Óbito:')
    print(melhor_linha.to_string())
    print(f'\nROC AUC: {auc:.4f}\n')
    print(relatorio)

    tabela_thresholds.to_csv(f'{PASTA_SAIDA}/thresholds_rf_completo.csv', sep=';', index=False)

    with open(f'{PASTA_SAIDA}/relatorio_rf_completo.txt', 'w', encoding='utf-8') as arquivo:
        arquivo.write('Features usadas:\n')
        arquivo.write('\n'.join(features))
        arquivo.write('\n\nDistribuição da variável alvo:\n')
        arquivo.write(y.value_counts().sort_index().to_string())
        arquivo.write('\n\nMelhor threshold por F1 da classe Óbito:\n')
        arquivo.write(melhor_linha.to_string())
        arquivo.write(f'\n\nROC AUC: {auc:.4f}\n\n')
        arquivo.write(relatorio)

    salvar_matriz_confusao(y_test, y_pred)
    salvar_curva_roc(y_test, y_prob)
    salvar_curva_precision_recall(y_test, y_prob)
    salvar_importancia_features(modelo, features)

    print(f'Arquivos salvos em: {PASTA_SAIDA}/')


if __name__ == '__main__':
    main()
