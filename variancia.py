from sklearn.feature_selection import VarianceThreshold
import pandas as pd

df = pd.read_csv('datasets/DATASETFINAL.csv', sep=';')

# 1. Supondo que 'X' seja o seu DataFrame contendo APENAS as variáveis preditoras 
# (Certifique-se de retirar a coluna de target 'EVOLUCAO' antes desse passo)
features = [
    'NOSOCOMIAL', 'AVE_SUINO', 'FEBRE', 'TOSSE', 'GARGANTA', 'DISPNEIA', 
    'DESC_RESP', 'SATURACAO', 'DIARREIA', 'VOMITO', 'OUTRO_SIN',
    'FATOR_RISC', 'PUERPERA', 'CARDIOPATI', 'HEMATOLOGI', 'SIND_DOWN', 
    'HEPATICA', 'ASMA', 'DIABETES', 'NEUROLOGIC', 'PNEUMOPATI', 'IMUNODEPRE', 
    'RENAL', 'OBESIDADE', 'OUT_MORBI', 'ANTIVIRAL', 'UTI', 'HOSPITAL', 'SUPORT_VEN'
]
X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)

# 2. Instancia o seletor com o limite de 0.01 exigido pelo professor
seletor = VarianceThreshold(threshold=0.01)

# 3. Treina o seletor nos dados
seletor.fit(X)

# 4. Cria um DataFrame para visualizar a variância de cada coluna
variancias = pd.DataFrame({
    'Atributo': X.columns,
    'Variância': seletor.variances_
}).sort_values(by='Variância', ascending=False)

print("--- VARIÂNCIA DE CADA ATRIBUTO ---")
print(variancias)

# 5. Descobre quais colunas serão eliminadas (variância <= 0.01)
colunas_removidas = X.columns[~seletor.get_support()]
print("\n--- COLUNAS ELIMINADAS (<= 0.01) ---")
print(colunas_removidas)

# 6. Cria o novo conjunto de dados filtrado apenas com as colunas aprovadas
X_filtrado = X.loc[:, seletor.get_support()]