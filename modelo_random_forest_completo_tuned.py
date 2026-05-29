import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    make_scorer,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

DATASET = 'datasets/DATASETFINAL_COMPLETO.csv'
PASTA_SAIDA = 'graficos_modelo_rf_completo_tuned'
ALVO = 'OBITO'
FEATURES_SELECIONADAS = [
    'SUPORT_VEN', 'IDADE_ANOS', 'UTI', 'CLASSI_FIN', 'FATOR_RISC',
    'VACINA_COV', 'HOSPITAL', 'SATURACAO', 'TOSSE', 'CARDIOPATI',
    'PCR_SARS2', 'CS_ESCOL_N'
]

PARAM_DIST = {
    'n_estimators': randint(100, 600),
    'max_depth': [6, 8, 10, 12, 14, 16, 18, None],
    'min_samples_split': randint(2, 25),
    'min_samples_leaf': randint(1, 15),
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
}

N_ITER = 40
CV_FOLDS = 3
SCORING = 'roc_auc'
# Fração do treino usada na busca de hiperparâmetros (dataset tem ~4M linhas)
# O modelo final é retreinado no treino completo após a busca.
SAMPLE_FRAC_TUNING = 0.10


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
        matriz, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Cura', 'Óbito'],
        yticklabels=['Cura', 'Óbito'],
    )
    plt.title('Matriz de Confusão - RF Tuned')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/matriz_confusao_rf_tuned.png', dpi=150)
    plt.close()


def salvar_curva_roc(y_real, y_prob):
    fpr, tpr, _ = roc_curve(y_real, y_prob)
    auc = roc_auc_score(y_real, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Aleatório')
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Curva ROC - RF Tuned')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/curva_roc_rf_tuned.png', dpi=150)
    plt.close()


def salvar_curva_precision_recall(y_real, y_prob):
    precision, recall, _ = precision_recall_curve(y_real, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color='steelblue', lw=2)
    plt.xlabel('Recall - Óbito')
    plt.ylabel('Precision - Óbito')
    plt.title('Curva Precision-Recall - RF Tuned')
    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/curva_precision_recall_rf_tuned.png', dpi=150)
    plt.close()


def salvar_importancia_features(modelo, features):
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    features_ordenadas = [features[i] for i in indices]
    valores_ordenados = importancias[indices]

    plt.figure(figsize=(10, 10))
    barras = plt.barh(features_ordenadas[::-1], valores_ordenados[::-1], color='steelblue')
    plt.xlabel('Importância')
    plt.title('Importância das Features - RF Tuned')

    for barra, valor in zip(barras, valores_ordenados[::-1]):
        plt.text(
            barra.get_width() + 0.002,
            barra.get_y() + barra.get_height() / 2,
            f'{valor:.3f}',
            va='center', fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/feature_importance_rf_tuned.png', dpi=150)
    plt.close()


def salvar_resultados_cv(search):
    resultados = pd.DataFrame(search.cv_results_)
    colunas = ['rank_test_score', 'mean_test_score', 'std_test_score',
               'param_n_estimators', 'param_max_depth', 'param_min_samples_split',
               'param_min_samples_leaf', 'param_max_features']
    resultados[colunas].sort_values('rank_test_score').to_csv(
        f'{PASTA_SAIDA}/cv_results.csv', sep=';', index=False
    )


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    df = pd.read_csv(DATASET, sep=';')

    colunas_ausentes = [c for c in FEATURES_SELECIONADAS + [ALVO] if c not in df.columns]
    if colunas_ausentes:
        raise ValueError(f'Colunas ausentes: {colunas_ausentes}')

    features = FEATURES_SELECIONADAS
    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(-1)
    y = pd.to_numeric(df[ALVO], errors='coerce')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    print(f'Tamanho treino: {len(X_train)}  |  Tamanho teste: {len(X_test)}')

    # Amostra estratificada do treino para a busca de hiperparâmetros
    idx_amostra = (
        pd.Series(y_train.values, index=y_train.index)
        .groupby(y_train.values)
        .apply(lambda g: g.sample(frac=SAMPLE_FRAC_TUNING, random_state=42))
        .droplevel(0)
        .index
    )
    X_tune = X_train.loc[idx_amostra]
    y_tune = y_train.loc[idx_amostra]

    print(f'Amostra para tuning: {len(X_tune)} ({SAMPLE_FRAC_TUNING*100:.0f}% do treino)')
    print(f'Distribuição alvo na amostra:\n{y_tune.value_counts().sort_index()}\n')
    print(f'Iniciando RandomizedSearchCV: {N_ITER} iterações, {CV_FOLDS}-fold CV, scoring={SCORING}\n')

    base_modelo = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_modelo,
        param_distributions=PARAM_DIST,
        n_iter=N_ITER,
        scoring=SCORING,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        refit=False,
    )
    search.fit(X_tune, y_tune)

    print('\nReetreinando modelo final com melhores parâmetros no treino COMPLETO...')
    modelo = RandomForestClassifier(
        **search.best_params_,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    print('\n--- Melhores hiperparâmetros encontrados ---')
    for param, valor in search.best_params_.items():
        print(f'  {param}: {valor}')
    print(f'  CV {SCORING} médio: {search.best_score_:.4f}\n')

    y_prob = modelo.predict_proba(X_test)[:, 1]
    tabela_thresholds = avaliar_thresholds(y_test, y_prob)
    melhor_linha = tabela_thresholds.sort_values('f1_obito', ascending=False).iloc[0]
    melhor_threshold = melhor_linha['threshold']
    y_pred = (y_prob >= melhor_threshold).astype(int)

    relatorio = classification_report(
        y_test, y_pred,
        target_names=['Cura (0)', 'Óbito (1)'],
        zero_division=0,
    )
    auc = roc_auc_score(y_test, y_prob)

    print('Distribuição da variável alvo:')
    print(y.value_counts().sort_index())
    print('\nMelhor threshold por F1 da classe Óbito:')
    print(melhor_linha.to_string())
    print(f'\nROC AUC: {auc:.4f}\n')
    print(relatorio)

    tabela_thresholds.to_csv(f'{PASTA_SAIDA}/thresholds_rf_tuned.csv', sep=';', index=False)
    salvar_resultados_cv(search)

    with open(f'{PASTA_SAIDA}/relatorio_rf_tuned.txt', 'w', encoding='utf-8') as arquivo:
        arquivo.write('Melhores hiperparâmetros (RandomizedSearchCV):\n')
        for param, valor in search.best_params_.items():
            arquivo.write(f'  {param}: {valor}\n')
        arquivo.write(f'\nCV {SCORING} médio: {search.best_score_:.4f}\n\n')
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

    print(f'\nArquivos salvos em: {PASTA_SAIDA}/')


if __name__ == '__main__':
    main()
