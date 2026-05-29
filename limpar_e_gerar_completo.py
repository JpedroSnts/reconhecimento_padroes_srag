import glob
import os

import numpy as np
import pandas as pd

CAMINHO_PASTA = 'datasets'
PADRAO_ARQUIVOS = os.path.join(CAMINHO_PASTA, 'INFLUD*.csv')
CAMINHO_FINAL = os.path.join(CAMINHO_PASTA, 'DATASETFINAL_COMPLETO.csv')

COLUNAS_DEMOGRAFICAS = [
    'NU_IDADE_N', 'TP_IDADE', 'CS_ESCOL_N'
]

COLUNAS_BINARIAS = [
    'UTI', 'FATOR_RISC', 'VACINA_COV', 'HOSPITAL', 'SATURACAO',
    'TOSSE', 'CARDIOPATI'
]

COLUNAS_CATEGORICAS = [
    'SUPORT_VEN', 'CLASSI_FIN', 'PCR_SARS2', 'CS_ESCOL_N'
]

COLUNAS_DESEJADAS = list(dict.fromkeys(
    COLUNAS_DEMOGRAFICAS
    + COLUNAS_BINARIAS
    + COLUNAS_CATEGORICAS
    + ['EVOLUCAO']
))


def mapear_binaria(serie):
    valores = pd.to_numeric(serie, errors='coerce')
    return valores.map({1: 1, 2: 0}).fillna(-1).astype(int)


def mapear_categorica(serie):
    valores = pd.to_numeric(serie, errors='coerce')
    return valores.where(~valores.isin([9, 99]), -1).fillna(-1).astype(int)


def corrigir_idade(df):
    df['TP_IDADE'] = pd.to_numeric(df['TP_IDADE'], errors='coerce')
    df['NU_IDADE_N'] = pd.to_numeric(df['NU_IDADE_N'], errors='coerce')

    idade_anos = df['NU_IDADE_N'].astype(float).copy()
    idade_anos.loc[df['TP_IDADE'] == 1] = idade_anos.loc[df['TP_IDADE'] == 1] / 365
    idade_anos.loc[df['TP_IDADE'] == 2] = idade_anos.loc[df['TP_IDADE'] == 2] / 12

    df['IDADE_ANOS'] = idade_anos.abs()
    df = df[(df['IDADE_ANOS'].notna()) & (df['IDADE_ANOS'] <= 110)].copy()

    return df.drop(columns=['NU_IDADE_N', 'TP_IDADE'])


def corrigir_sexo(df):
    sexo = df['CS_SEXO'].astype(str).str.upper().str.strip()
    df['CS_SEXO'] = sexo.map({
        'M': 0, '1': 0, '1.0': 0,
        'F': 1, '2': 1, '2.0': 1,
    }).fillna(-1).astype(int)
    return df


def main():
    arquivos = sorted(glob.glob(PADRAO_ARQUIVOS))

    if not arquivos:
        raise FileNotFoundError(f'Nenhum arquivo encontrado em {PADRAO_ARQUIVOS}')

    print('Lendo e unificando arquivos INFLUD...')
    df = pd.concat(
        [
            pd.read_csv(
                arquivo,
                sep=';',
                usecols=COLUNAS_DESEJADAS,
                encoding='latin-1',
                low_memory=False,
            )
            for arquivo in arquivos
        ],
        ignore_index=True,
    )

    print('Filtrando evolução para Cura e Óbito...')
    df['EVOLUCAO'] = pd.to_numeric(df['EVOLUCAO'], errors='coerce')
    df = df[df['EVOLUCAO'].isin([1, 2])].copy()
    df['OBITO'] = df['EVOLUCAO'].map({1: 0, 2: 1}).astype(int)
    df = df.drop(columns=['EVOLUCAO'])

    print('Corrigindo idade...')
    df = corrigir_idade(df)

    print('Mapeando variáveis binárias e categóricas...')
    for coluna in COLUNAS_BINARIAS:
        df[coluna] = mapear_binaria(df[coluna])

    for coluna in COLUNAS_CATEGORICAS:
        df[coluna] = mapear_categorica(df[coluna])

    colunas_finais = [
        'SUPORT_VEN', 'IDADE_ANOS', 'UTI', 'CLASSI_FIN', 'FATOR_RISC',
        'VACINA_COV', 'HOSPITAL', 'SATURACAO', 'TOSSE', 'CARDIOPATI',
        'PCR_SARS2', 'OBITO', 'CS_ESCOL_N'
    ]
    colunas_finais = list(dict.fromkeys(colunas_finais))
    df = df[colunas_finais]

    os.makedirs(CAMINHO_PASTA, exist_ok=True)
    df.to_csv(CAMINHO_FINAL, sep=';', index=False)

    print('Dataset completo gerado com sucesso!')
    print(f'Arquivo: {CAMINHO_FINAL}')
    print(f'Formato: {df.shape}')
    print('Distribuição do alvo:')
    print(df['OBITO'].value_counts().sort_index())


if __name__ == '__main__':
    main()
