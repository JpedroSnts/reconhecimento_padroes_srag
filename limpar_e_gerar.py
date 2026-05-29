import pandas as pd
import glob
import os
import numpy as np
from ydata_profiling import ProfileReport

caminho_pasta = 'datasets'
arquivos = glob.glob(os.path.join(caminho_pasta, "INFLUD*.csv"))

# OTIMIZAÇÃO: Mantivemos apenas as colunas que seu Random Forest realmente usa + colunas de controle
colunas_desejadas = [
    'CS_SEXO', 'NU_IDADE_N', 'TP_IDADE', 'PUERPERA', 'CARDIOPATI', 
    'HEMATOLOGI', 'SIND_DOWN', 'HEPATICA', 'ASMA', 'DIABETES', 
    'NEUROLOGIC', 'PNEUMOPATI', 'IMUNODEPRE', 'RENAL', 'OBESIDADE',
    'OUT_MORBI', 'NOSOCOMIAL', 'FEBRE', 'TOSSE', 'GARGANTA', 
    'DISPNEIA', 'DESC_RESP', 'SATURACAO', 'DIARREIA', 'VOMITO', 
    'OUTRO_SIN', 'ANTIVIRAL', 'HOSPITAL', 'UTI', 'SUPORT_VEN', 'EVOLUCAO',
    'FADIGA', 'PERD_OLFT', 'PERD_PALA', 'VACINA', 'VACINA_COV'
]

print("Lendo e unificando os arquivos da pasta...")
df_filtrado = pd.concat(
    [pd.read_csv(f, sep=";", usecols=colunas_desejadas, encoding='latin-1', low_memory=False) for f in arquivos], 
    ignore_index=True
)

# Filtra a coluna EVOLUCAO para manter apenas 1 (Cura) e 2 (Óbito)
df_filtrado = df_filtrado[df_filtrado['EVOLUCAO'].isin([1, 2, '1', '2', 1.0, 2.0])]

print("Processando e corrigindo variáveis...")

# 1. CORREÇÃO DA IDADE (A armadilha do TP_IDADE)
df_filtrado['TP_IDADE'] = pd.to_numeric(df_filtrado['TP_IDADE'], errors='coerce')
df_filtrado['NU_IDADE_N'] = pd.to_numeric(df_filtrado['NU_IDADE_N'], errors='coerce')

# Converte dias e meses para frações de ano de forma correta
df_filtrado.loc[df_filtrado['TP_IDADE'] == 1, 'NU_IDADE_N'] = df_filtrado['NU_IDADE_N'] / 365
df_filtrado.loc[df_filtrado['TP_IDADE'] == 2, 'NU_IDADE_N'] = df_filtrado['NU_IDADE_N'] / 12
df_filtrado = df_filtrado.drop(columns=['TP_IDADE'])

# Filtros físicos de idade
df_filtrado = df_filtrado[df_filtrado['NU_IDADE_N'] <= 110]
df_filtrado['NU_IDADE_N'] = df_filtrado['NU_IDADE_N'].abs()


# 2. CORREÇÃO DO CS_SEXO (Evita que a variável vire tudo zero no to_numeric)
# Padroniza para string limpa e mapeia: Masculino = 0, Feminino = 1. Ignorados viram NaN.
df_filtrado['CS_SEXO'] = df_filtrado['CS_SEXO'].astype(str).str.upper().str.strip()
df_filtrado['CS_SEXO'] = df_filtrado['CS_SEXO'].map({'M': 0, '1': 0, '1.0': 0, 'F': 1, '2': 1, '2.0': 1})


# 3. MAPEAMENTO CLÍNICO (Isolando os ignorados '9' e vazios como NaN para exclusão futura)
colunas_clinicas = [
   'PUERPERA', 'CARDIOPATI', 'HEMATOLOGI', 'SIND_DOWN', 'HEPATICA', 
    'ASMA', 'DIABETES', 'NEUROLOGIC', 'PNEUMOPATI', 'IMUNODEPRE', 
    'RENAL', 'OBESIDADE', 'OUT_MORBI', 'NOSOCOMIAL', 'FEBRE', 'TOSSE', 
    'GARGANTA', 'DISPNEIA', 'DESC_RESP', 'SATURACAO', 'DIARREIA', 
    'VOMITO', 'OUTRO_SIN', 'ANTIVIRAL', 'HOSPITAL', 'UTI', 
    'PERD_OLFT', 'FADIGA', 'PERD_PALA', 'VACINA', 'VACINA_COV'
]

for col in colunas_clinicas:
    df_filtrado[col] = pd.to_numeric(df_filtrado[col], errors='coerce')
    # 1 vira 1 (Sim), 2 vira 0 (Não). O código 9 (Ignorado) e brancos viram NaN automaticamente.
    df_filtrado[col] = df_filtrado[col].map({1: 1, 2: 0})


# 4. CORREÇÃO DO SUPORT_VEN (Inteligência clínica preservada)
df_filtrado['SUPORT_VEN'] = pd.to_numeric(df_filtrado['SUPORT_VEN'], errors='coerce')
df_filtrado['SUPORT_VEN'] = df_filtrado['SUPORT_VEN'].map({1: 1, 2: 1, 3: 0})


# 5. EXCLUSÃO DIRETA DOS CASOS INCOMPLETOS (A sua escolha metodológica)
# Como mapeamos tudo o que era '9' ou vazio para NaN, o dropna vai eliminar exatamente essas linhas ruins.
colunas_para_validar = colunas_clinicas + ['CS_SEXO', 'SUPORT_VEN']
df_filtrado = df_filtrado.dropna(subset=colunas_para_validar)


# 6. CRIAÇÃO DA FAIXA ETÁRIA (Feita após o dropna para evitar distorções)
limites_idade = [0, 5, 9, 19, 59, float('inf')]
categorias_idade = [1, 2, 3, 4, 5]

df_filtrado['FAIXA_ETARIA'] = pd.cut(
    df_filtrado['NU_IDADE_N'], 
    bins=limites_idade, 
    labels=categorias_idade, 
    include_lowest=True
).astype(int)

# Removemos a idade bruta porque o modelo usará apenas a FAIXA_ETARIA categórica
df_filtrado = df_filtrado.drop(columns=['NU_IDADE_N'])


# 7. MAPEAMENTO DA VARIÁVEL ALVO (OBITO)
df_filtrado['EVOLUCAO'] = pd.to_numeric(df_filtrado['EVOLUCAO'], errors='coerce')
df_filtrado['OBITO'] = df_filtrado['EVOLUCAO'].map({1: 0, 2: 1})
df_filtrado = df_filtrado.drop(columns=['EVOLUCAO'])


# 8. SALVAMENTO DO DATASET
print("Salvando o novo dataset final...")
os.makedirs('datasets', exist_ok=True)

print("Dataset filtrado e unificado com sucesso! Tamanho final das observações completas:", df_filtrado.shape)

caminho_final = 'datasets/DATASETFINAL.csv'
df_filtrado.to_csv(caminho_final, sep=';', index=False)

print(f"Sucesso! Seu novo dataset está limpo, padronizado e salvo como '{caminho_final}'.")


ProfileReport(df_filtrado, title="Relatório - DATASETFINAL", minimal=True).to_file("relatorio_binarizado.html")