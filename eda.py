import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ──────────────────────────────────────────────
# CONFIGURAÇÕES GERAIS
# ──────────────────────────────────────────────
df = pd.read_csv('datasets/DATASETFINAL.csv', sep=';')

pasta = 'graficos_eda'
os.makedirs(pasta, exist_ok=True)

# Mapeamentos
faixa_labels = {
    1: '0-5 anos',
    2: '6-9 anos',
    3: '10-19 anos',
    4: '20-59 anos',
    5: '60+ anos'
}
sexo_labels = {'M': 'Masculino', 'F': 'Feminino'}  # CS_SEXO é string no CSV

# Tipos corretos
df['FAIXA_ETARIA'] = pd.to_numeric(df['FAIXA_ETARIA'], errors='coerce')
df['NU_IDADE_N']   = pd.to_numeric(df['NU_IDADE_N'],   errors='coerce')
df['OBITO']        = pd.to_numeric(df['OBITO'],         errors='coerce')
df['CS_SEXO']      = df['CS_SEXO'].astype(str).str.strip().str.upper()

# Colunas legíveis para os gráficos
df['FAIXA_LABEL'] = df['FAIXA_ETARIA'].map(faixa_labels)
df['SEXO_LABEL']  = df['CS_SEXO'].map(sexo_labels)  # 'I' vira NaN, ignorado

ordem_faixas = list(faixa_labels.values())
cores_sexo   = {'Masculino': '#2E86AB', 'Feminino': '#E84855'}

# ──────────────────────────────────────────────
# GRÁFICO 1 — Boxplot: Distribuição de Idade por Faixa Etária x Sexo
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

df_box = df.dropna(subset=['FAIXA_LABEL', 'SEXO_LABEL', 'NU_IDADE_N'])

sns.boxplot(
    data=df_box,
    x='FAIXA_LABEL',
    y='NU_IDADE_N',
    hue='SEXO_LABEL',
    order=ordem_faixas,
    hue_order=['Masculino', 'Feminino'],
    palette=cores_sexo,
    width=0.55,
    linewidth=1.2,
    flierprops=dict(marker='o', markersize=3, alpha=0.4),
    ax=ax
)

ax.set_title('Distribuição de Idade por Faixa Etária e Sexo', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Faixa Etária', fontsize=11)
ax.set_ylabel('Idade (anos)', fontsize=11)
handles = [mpatches.Patch(color=cores_sexo[s], label=s) for s in ['Masculino', 'Feminino']]
ax.legend(handles=handles, title='Sexo', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)
sns.despine()

plt.tight_layout()
plt.savefig(f'{pasta}/boxplot_faixa_sexo.png', dpi=150)

# ──────────────────────────────────────────────
# GRÁFICO 2 — Colunas: Número de Casos por Faixa Etária x Sexo
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

df_col = df.dropna(subset=['FAIXA_LABEL', 'SEXO_LABEL'])

contagem = (
    df_col.groupby(['FAIXA_LABEL', 'SEXO_LABEL'])
    .size()
    .reset_index(name='TOTAL')
)

sns.barplot(
    data=contagem,
    x='FAIXA_LABEL',
    y='TOTAL',
    hue='SEXO_LABEL',
    order=ordem_faixas,
    hue_order=['Masculino', 'Feminino'],
    palette=cores_sexo,
    ax=ax
)

for container in ax.containers:
    ax.bar_label(container, fmt='%d', fontsize=8, padding=3)

ax.set_title('Número de Casos por Faixa Etária e Sexo', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Faixa Etária', fontsize=11)
ax.set_ylabel('Quantidade de Pacientes', fontsize=11)
handles = [mpatches.Patch(color=cores_sexo[s], label=s) for s in ['Masculino', 'Feminino']]
ax.legend(handles=handles, title='Sexo', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)
sns.despine()

plt.tight_layout()
plt.savefig(f'{pasta}/colunas_faixa_sexo.png', dpi=150)

# ──────────────────────────────────────────────
# GRÁFICO 3 — Matriz de Correlação: Features x Óbito
# ──────────────────────────────────────────────
features_corr = [
    'FAIXA_ETARIA', 'PUERPERA', 'CARDIOPATI', 'HEMATOLOGI', 'SIND_DOWN', 
    'HEPATICA', 'ASMA', 'DIABETES', 'NEUROLOGIC', 'PNEUMOPATI', 
    'IMUNODEPRE', 'RENAL', 'OBESIDADE', 'OUT_MORBI', 'NOSOCOMIAL',
    'FEBRE', 'TOSSE', 'GARGANTA', 'DISPNEIA', 'DESC_RESP', 'SATURACAO',
    'DIARREIA', 'VOMITO', 'OUTRO_SIN', 'FATOR_RISC',
    'ANTIVIRAL', 'HOSPITAL', 'UTI', 'SUPORT_VEN', 'OBITO'
]

features_presentes = [f for f in features_corr if f in df.columns]

df_corr = df[features_presentes].apply(pd.to_numeric, errors='coerce').fillna(0)
matriz  = df_corr.corr()

# Ordena pela correlação com ÓBITO (maior → menor)
corr_obito = matriz['OBITO'].drop('OBITO').sort_values(ascending=False)
ordem_corr = corr_obito.index.tolist() + ['OBITO']
matriz_ord = matriz.loc[ordem_corr, ordem_corr]

fig, ax = plt.subplots(figsize=(14, 12))

mask = np.zeros_like(matriz_ord, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True  # oculta triângulo superior

sns.heatmap(
    matriz_ord,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.4,
    annot_kws={'size': 7},
    ax=ax
)

ax.set_title('Matriz de Correlação — Features com Óbito', fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(f'{pasta}/matriz_correlacao.png', dpi=150)
