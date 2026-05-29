import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

DATASET = 'datasets/DATASETFINAL_COMPLETO.csv'
PASTA = 'graficos_eda_completo'

BINARIAS = ['UTI', 'FATOR_RISC', 'VACINA_COV', 'HOSPITAL', 'SATURACAO', 'TOSSE', 'CARDIOPATI']
CATEGORICAS = ['SUPORT_VEN', 'CLASSI_FIN', 'PCR_SARS2', 'CS_ESCOL_N']
FEATURES_MODELO = BINARIAS + CATEGORICAS + ['IDADE_ANOS']

LABELS_SUPORT_VEN = {1: 'Sim, invasivo', 2: 'Sim, não invasivo', 3: 'Não', -1: 'Ignorado'}
LABELS_CLASSI_FIN = {1: 'SRAG por Influenza', 2: 'SRAG por outro vírus', 3: 'SRAG por outro agente',
                     4: 'Não especificado', 5: 'COVID-19', -1: 'Ignorado'}
LABELS_PCR        = {1: 'Positivo', 2: 'Negativo', 3: 'Inconclusivo', -1: 'Ignorado'}
LABELS_ESCOL      = {0: 'Sem escolaridade', 1: 'Fund. incompleto', 2: 'Fund. completo',
                     3: 'Médio incompleto', 4: 'Médio completo', 5: 'Superior incompleto',
                     6: 'Superior completo', -1: 'Ignorado'}
LABELS_BINARIA    = {1: 'Sim', 0: 'Não', -1: 'Ignorado'}

COR_CURA  = '#2E86AB'
COR_OBITO = '#E84855'


def carregar():
    df = pd.read_csv(DATASET, sep=';')
    df['OBITO'] = pd.to_numeric(df['OBITO'], errors='coerce')
    df['IDADE_ANOS'] = pd.to_numeric(df['IDADE_ANOS'], errors='coerce')
    for col in BINARIAS + CATEGORICAS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)
    return df


# ── Gráfico 1: Distribuição da variável alvo ────────────────────────────────
def grafico_distribuicao_alvo(df):
    contagem = df['OBITO'].value_counts().sort_index()
    labels = ['Cura', 'Óbito']
    cores = [COR_CURA, COR_OBITO]
    total = contagem.sum()

    fig, ax = plt.subplots(figsize=(7, 5))
    barras = ax.bar(labels, contagem.values, color=cores, edgecolor='white', width=0.5)

    for barra, valor in zip(barras, contagem.values):
        ax.text(barra.get_x() + barra.get_width() / 2,
                barra.get_height() + total * 0.005,
                f'{valor:,}\n({valor/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=11)

    ax.set_title('Distribuição da Variável Alvo (ÓBITO)', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Número de Registros', fontsize=11)
    ax.set_ylim(0, contagem.max() * 1.18)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{PASTA}/01_distribuicao_alvo.png', dpi=150)
    plt.close()


# ── Gráfico 2: Distribuição de idade por desfecho ───────────────────────────
def grafico_idade_por_desfecho(df):
    df_idade = df[df['IDADE_ANOS'].notna() & df['OBITO'].notna()].copy()
    df_idade['Desfecho'] = df_idade['OBITO'].map({0: 'Cura', 1: 'Óbito'})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(data=df_idade, x='Desfecho', y='IDADE_ANOS',
                palette={'Cura': COR_CURA, 'Óbito': COR_OBITO},
                order=['Cura', 'Óbito'], ax=axes[0], width=0.5)
    axes[0].set_title('Boxplot de Idade por Desfecho', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Idade (anos)')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine(ax=axes[0])

    for desfecho, cor in [('Cura', COR_CURA), ('Óbito', COR_OBITO)]:
        dados = df_idade[df_idade['Desfecho'] == desfecho]['IDADE_ANOS']
        axes[1].hist(dados, bins=40, alpha=0.6, color=cor, label=desfecho, edgecolor='white')
    axes[1].set_title('Distribuição de Idade por Desfecho', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Idade (anos)')
    axes[1].set_ylabel('Frequência')
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine(ax=axes[1])

    plt.suptitle('Idade × Desfecho', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{PASTA}/02_idade_por_desfecho.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Gráfico 3: Taxa de óbito por variável binária ───────────────────────────
def grafico_taxa_obito_binarias(df):
    linhas = []
    for col in BINARIAS:
        for valor, label in [(1, 'Sim'), (0, 'Não')]:
            sub = df[(df[col] == valor) & df['OBITO'].notna()]
            if len(sub) == 0:
                continue
            taxa = sub['OBITO'].mean() * 100
            linhas.append({'Feature': col, 'Categoria': label, 'Taxa Óbito (%)': taxa, 'N': len(sub)})

    df_taxa = pd.DataFrame(linhas)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_taxa, x='Feature', y='Taxa Óbito (%)', hue='Categoria',
                hue_order=['Sim', 'Não'],
                palette={'Sim': COR_OBITO, 'Não': COR_CURA}, ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=8, padding=3)

    ax.set_title('Taxa de Óbito por Variável Binária (Sim vs Não)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('Taxa de Óbito (%)')
    ax.set_ylim(0, df_taxa['Taxa Óbito (%)'].max() * 1.2)
    ax.legend(title='Valor', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{PASTA}/03_taxa_obito_binarias.png', dpi=150)
    plt.close()


# ── Gráfico 4: Matriz de correlação ─────────────────────────────────────────
def grafico_correlacao(df):
    cols = FEATURES_MODELO + ['OBITO']
    df_corr = df[cols].apply(pd.to_numeric, errors='coerce').fillna(-1)
    matriz = df_corr.corr()

    corr_obito = matriz['OBITO'].drop('OBITO').sort_values(ascending=False)
    ordem = corr_obito.index.tolist() + ['OBITO']
    matriz_ord = matriz.loc[ordem, ordem]

    mask = np.zeros_like(matriz_ord, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matriz_ord, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                linewidths=0.4, annot_kws={'size': 8}, ax=ax)

    ax.set_title('Matriz de Correlação — Features × Óbito\n(ordenada por correlação com desfecho)',
                 fontsize=13, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{PASTA}/04_matriz_correlacao.png', dpi=150)
    plt.close()


# ── Gráfico 5: Suporte ventilatório × desfecho ──────────────────────────────
def grafico_suport_ven(df):
    df_s = df[df['OBITO'].notna() & (df['SUPORT_VEN'] != -1)].copy()
    df_s['Suporte'] = df_s['SUPORT_VEN'].map(LABELS_SUPORT_VEN)
    df_s['Desfecho'] = df_s['OBITO'].map({0: 'Cura', 1: 'Óbito'})

    contagem = (df_s.groupby(['Suporte', 'Desfecho']).size()
                .reset_index(name='N'))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=contagem, x='Suporte', y='N', hue='Desfecho',
                hue_order=['Cura', 'Óbito'],
                palette={'Cura': COR_CURA, 'Óbito': COR_OBITO}, ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt='%d', fontsize=8, padding=3)

    ax.set_title('Suporte Ventilatório por Desfecho', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Tipo de Suporte')
    ax.set_ylabel('Número de Registros')
    ax.legend(title='Desfecho', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{PASTA}/05_suport_ven_desfecho.png', dpi=150)
    plt.close()


# ── Gráfico 6: Classificação final × taxa de óbito ──────────────────────────
def grafico_classi_fin(df):
    df_c = df[df['OBITO'].notna() & (df['CLASSI_FIN'] != -1)].copy()
    df_c['Classificação'] = df_c['CLASSI_FIN'].map(LABELS_CLASSI_FIN)

    taxa = (df_c.groupby('Classificação')['OBITO']
            .agg(['mean', 'count'])
            .reset_index()
            .rename(columns={'mean': 'Taxa', 'count': 'N'}))
    taxa['Taxa (%)'] = taxa['Taxa'] * 100
    taxa = taxa.sort_values('Taxa (%)', ascending=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    barras = ax.bar(taxa['Classificação'], taxa['Taxa (%)'],
                    color=COR_OBITO, edgecolor='white', alpha=0.85)

    for barra, (_, row) in zip(barras, taxa.iterrows()):
        ax.text(barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 0.3,
                f'{row["Taxa (%)"]:.1f}%\n(n={int(row["N"]):,})',
                ha='center', va='bottom', fontsize=8)

    ax.set_title('Taxa de Óbito por Classificação Final (CLASSI_FIN)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Taxa de Óbito (%)')
    ax.set_ylim(0, taxa['Taxa (%)'].max() * 1.25)
    plt.xticks(rotation=20, ha='right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{PASTA}/06_classi_fin_taxa_obito.png', dpi=150)
    plt.close()


# ── Gráfico 7: Vacinação COVID × desfecho ───────────────────────────────────
def grafico_vacina_cov(df):
    df_v = df[df['OBITO'].notna() & (df['VACINA_COV'] != -1)].copy()
    df_v['Vacinado'] = df_v['VACINA_COV'].map({1: 'Sim', 0: 'Não'})
    df_v['Desfecho'] = df_v['OBITO'].map({0: 'Cura', 1: 'Óbito'})

    total_por_vac = df_v.groupby('Vacinado').size()
    contagem = (df_v.groupby(['Vacinado', 'Desfecho']).size()
                .reset_index(name='N'))
    contagem['%'] = contagem.apply(
        lambda r: r['N'] / total_por_vac[r['Vacinado']] * 100, axis=1
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=contagem, x='Vacinado', y='N', hue='Desfecho',
                hue_order=['Cura', 'Óbito'],
                palette={'Cura': COR_CURA, 'Óbito': COR_OBITO}, ax=axes[0])
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%d', fontsize=9, padding=3)
    axes[0].set_title('Volume por Vacinação e Desfecho', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Vacinado contra COVID-19')
    axes[0].set_ylabel('Registros')
    axes[0].legend(title='Desfecho')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine(ax=axes[0])

    sns.barplot(data=contagem, x='Vacinado', y='%', hue='Desfecho',
                hue_order=['Cura', 'Óbito'],
                palette={'Cura': COR_CURA, 'Óbito': COR_OBITO}, ax=axes[1])
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.1f%%', fontsize=9, padding=3)
    axes[1].set_title('Proporção por Vacinação e Desfecho', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Vacinado contra COVID-19')
    axes[1].set_ylabel('% dentro do grupo')
    axes[1].legend(title='Desfecho')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine(ax=axes[1])

    plt.suptitle('Vacinação COVID-19 × Desfecho', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{PASTA}/07_vacina_cov_desfecho.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Gráfico 8: Escolaridade × taxa de óbito ─────────────────────────────────
def grafico_escolaridade(df):
    df_e = df[df['OBITO'].notna() & (df['CS_ESCOL_N'] != -1)].copy()
    df_e['Escolaridade'] = df_e['CS_ESCOL_N'].map(LABELS_ESCOL)

    taxa = (df_e.groupby('Escolaridade')['OBITO']
            .agg(['mean', 'count'])
            .reset_index()
            .rename(columns={'mean': 'Taxa', 'count': 'N'}))
    taxa['Taxa (%)'] = taxa['Taxa'] * 100

    ordem = [LABELS_ESCOL[k] for k in sorted(LABELS_ESCOL) if k != -1 and LABELS_ESCOL[k] in taxa['Escolaridade'].values]

    fig, ax = plt.subplots(figsize=(12, 5))
    barras = ax.bar(
        [o for o in ordem if o in taxa['Escolaridade'].values],
        [taxa.loc[taxa['Escolaridade'] == o, 'Taxa (%)'].values[0] for o in ordem if o in taxa['Escolaridade'].values],
        color=COR_OBITO, edgecolor='white', alpha=0.85
    )
    ns = [taxa.loc[taxa['Escolaridade'] == o, 'N'].values[0] for o in ordem if o in taxa['Escolaridade'].values]

    for barra, n in zip(barras, ns):
        ax.text(barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 0.2,
                f'{barra.get_height():.1f}%\n(n={int(n):,})',
                ha='center', va='bottom', fontsize=8)

    ax.set_title('Taxa de Óbito por Escolaridade (CS_ESCOL_N)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Taxa de Óbito (%)')
    ax.set_ylim(0, taxa['Taxa (%)'].max() * 1.25)
    plt.xticks(rotation=25, ha='right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{PASTA}/08_escolaridade_taxa_obito.png', dpi=150)
    plt.close()


def main():
    os.makedirs(PASTA, exist_ok=True)
    print('Carregando dataset...')
    df = carregar()
    print(f'Registros: {len(df):,}  |  Óbitos: {df["OBITO"].sum():,.0f} ({df["OBITO"].mean()*100:.1f}%)\n')

    print('Gerando gráficos...')
    grafico_distribuicao_alvo(df)
    print('  1/8 — Distribuição do alvo')
    grafico_idade_por_desfecho(df)
    print('  2/8 — Idade por desfecho')
    grafico_taxa_obito_binarias(df)
    print('  3/8 — Taxa de óbito por variáveis binárias')
    grafico_correlacao(df)
    print('  4/8 — Matriz de correlação')
    grafico_suport_ven(df)
    print('  5/8 — Suporte ventilatório')
    grafico_classi_fin(df)
    print('  6/8 — Classificação final')
    grafico_vacina_cov(df)
    print('  7/8 — Vacinação COVID-19')
    grafico_escolaridade(df)
    print('  8/8 — Escolaridade')

    print(f'\nGráficos salvos em: {PASTA}/')


if __name__ == '__main__':
    main()
