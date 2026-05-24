import pandas as pd
import glob
import os

caminho_pasta = 'datasets'

arquivos = glob.glob(os.path.join(caminho_pasta, "INFLUD*.csv"))

colunas_desejadas = [
    'CS_SEXO', 'NU_IDADE_N', 'CS_GESTANT', 'CS_RACA', 'CS_ZONA',
    'NOSOCOMIAL', 'AVE_SUINO', 'FEBRE', 'TOSSE', 'GARGANTA', 'DISPNEIA', 
    'DESC_RESP', 'SATURACAO', 'DIARREIA', 'VOMITO', 'OUTRO_SIN',
    'FATOR_RISC', 'PUERPERA', 'CARDIOPATI', 'HEMATOLOGI', 'SIND_DOWN', 
    'HEPATICA', 'ASMA', 'DIABETES', 'NEUROLOGIC', 'PNEUMOPATI', 'IMUNODEPRE', 
    'RENAL', 'OBESIDADE', 'OUT_MORBI', 'ANTIVIRAL', 'UTI', 'HOSPITAL', 'SUPORT_VEN', 'EVOLUCAO'
]

df_filtrado = pd.concat(
    [pd.read_csv(f, sep=";", usecols=colunas_desejadas, encoding='latin-1', low_memory=False) for f in arquivos], 
    ignore_index=True
)

# Filtra a coluna EVOLUCAO para manter apenas 1 (Cura) e 2 (Óbito)
# Usamos strings e inteiros na lista para evitar problemas de tipagem na leitura do CSV
df_filtrado = df_filtrado[df_filtrado['EVOLUCAO'].isin([1, 2, '1', '2'])]

colunas_1_2_9 = [
    'PUERPERA', 'CARDIOPATI', 'HEMATOLOGI', 'SIND_DOWN', 
    'HEPATICA', 'ASMA', 'DIABETES', 'NEUROLOGIC', 'PNEUMOPATI', 
    'IMUNODEPRE', 'RENAL', 'OBESIDADE', 'OUT_MORBI', 'NOSOCOMIAL',
    'FEBRE', 'TOSSE', 'GARGANTA', 'DISPNEIA', 'DESC_RESP', 'SATURACAO',
    'DIARREIA', 'VOMITO', 'OUTRO_SIN', 'FATOR_RISC', 'VACINA',
    'ANTIVIRAL', 'HOSPITAL', 'UTI'
]

# Aplica a regra: Se for 1 (numero ou string) vira 1. Qualquer outra coisa (2, 9, NaN) vira 0.
for col in colunas_1_2_9:
    df_filtrado[col] = df_filtrado[col].isin([1, '1', 1.0, '1.0']).astype(int)

# NOVO: Imputação de valores nulos pela Mediana
colunas_numericas = ['NU_IDADE_N']
for col in colunas_numericas:
    # Calcula a mediana da coluna desconsiderando os nulos automaticamente
    mediana = df_filtrado[col].median()
    # Preenche os vazios (NaN) com a mediana calculada
    df_filtrado[col] = df_filtrado[col].fillna(mediana)

# Transforma 1º, 2º e 3º trimestre em 1 (Sim). Todo o resto (4, 5, 6, 9 e NaN) vira 0 (Não).
df_filtrado['CS_GESTANT'] = df_filtrado['CS_GESTANT'].isin([1, 2, 3, '1', '2', '3', 1.0, 2.0, 3.0]).astype(int)

# O que for 1 vira 1, qualquer outra coisa ou vazio vira 0
df_filtrado['CS_ZONA'] = df_filtrado['CS_ZONA'].isin([1, '1', 1.0, '1.0']).astype(int)

# Transforma 1 (Aves/Suínos) em 1. Todo o resto (2, 3, 9, NaN) vira 0.
df_filtrado['AVE_SUINO'] = df_filtrado['AVE_SUINO'].isin([1, '1', 1.0, '1.0']).astype(int)

# Transforma 1, 2 em 1 (Sim). Todo o resto (3, 9 e NaN) vira 0 (Não).
df_filtrado['SUPORT_VEN'] = df_filtrado['SUPORT_VEN'].isin([1, 2, '1', '2', 1.0, 2.0]).astype(int)

# Remoção de linhas com idade maior que 110 anos
df_filtrado = df_filtrado[df_filtrado['NU_IDADE_N'] <= 110]

# Aplica valor absoluto para remover idades negativas
df_filtrado['NU_IDADE_N'] = df_filtrado['NU_IDADE_N'].abs()
# Definindo os limites (bins) e as categorias (labels)
limites_idade = [0, 5, 9, 19, 59, float('inf')] # float('inf') garante que qualquer idade acima de 59 entre no último grupo
categorias_idade = [1, 2, 3, 4, 5]

# Aplicando o pd.cut para criar a nova coluna categórica
# include_lowest=True garante que o valor 0 (bebês com menos de 1 ano) seja incluído na primeira faixa
df_filtrado['FAIXA_ETARIA'] = pd.cut(
    df_filtrado['NU_IDADE_N'], 
    bins=limites_idade, 
    labels=categorias_idade, 
    include_lowest=True
)
df_filtrado['FAIXA_ETARIA'] = df_filtrado['FAIXA_ETARIA'].astype(int)

# Garante que a coluna está como numérica para evitar erros no mapeamento
df_filtrado['EVOLUCAO'] = pd.to_numeric(df_filtrado['EVOLUCAO'], errors='coerce')

print("Criando a variável OBITO (0 = Cura, 1 = Óbito)...")
# Cria a nova coluna mapeando 1 (Cura) para 0, e 2 (Óbito) para 1
df_filtrado['OBITO'] = df_filtrado['EVOLUCAO'].map({1: 0, 2: 1})

print("Removendo a variável antiga EVOLUCAO...")
# Remove (drop) a coluna original para não ter dados duplicados ou causar Data Leakage
df_filtrado = df_filtrado.drop(columns=['EVOLUCAO'])

print("Salvando o novo dataset final...")
# Garante que a pasta datasets existe
os.makedirs('datasets', exist_ok=True)

print("Dataset filtrado e unificado com sucesso! Tamanho atual:", df_filtrado.shape)

# Salva o novo arquivo (index=False evita que o Pandas crie uma coluna extra de numeração)
caminho_final = 'datasets/DATASETFINAL.csv'
df_filtrado.to_csv(caminho_final, sep=';', index=False)

print(f"Sucesso! Seu novo dataset está limpo, padronizado e salvo como '{caminho_final}'.")