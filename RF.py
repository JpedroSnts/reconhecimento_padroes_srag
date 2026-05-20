import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Leitura dos Dados
df_filtrado = pd.read_csv('datasets/DATASETFINAL.csv', sep=';')

pasta_ml = 'graficos_ml'
os.makedirs(pasta_ml, exist_ok=True)

print("\n==========================================")
print("INICIANDO MODELAGEM PREDITIVA (RANDOM FOREST - 80/20)")
print("==========================================\n")

# 2. Seleção das Features (Fatores de Risco e Perfil Biológico)
# Incluindo CS_SEXO, NU_IDADE_N (ou FAIXA_ETARIA) conforme o Dicionário de Dados
features = [
    'NU_IDADE_N', 'FAIXA_ETARIA', 'CS_SEXO', # Perfil biológico
    'PUERPERA', 'CARDIOPATI', 'HEMATOLOGI', 'SIND_DOWN', 
    'HEPATICA', 'ASMA', 'DIABETES', 'NEUROLOGIC', 'PNEUMOPATI', 
    'IMUNODEPRE', 'RENAL', 'OBESIDADE', # Comorbidades
    'DISPNEIA', 'SATURACAO', 'DESC_RESP', # Sinais e Sintomas
    'UTI', 'SUPORT_VEN' # Intervenções
]

# Garantindo que apenas colunas presentes no dataset sejam usadas
features_presentes = [f for f in features if f in df_filtrado.columns]

X = df_filtrado[features_presentes].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df_filtrado['OBITO']

print(f"Total de pacientes para treinamento e teste: {len(X)}")

# 3. Divisão de Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f"Pacientes para a IA estudar (Treino 80%): {len(X_train)}")
print(f"Pacientes para a prova final (Teste 20%): {len(X_test)}")

# 4. Construção e Treinamento do Modelo
print("\nTreinando a Floresta Aleatória...")
rf_model = RandomForestClassifier(
    n_estimators=200,      
    max_depth=10,          
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1              
)
rf_model.fit(X_train, y_train)

# 5. Avaliação
print("\nAplicando previsões na base de Teste...")
y_pred = rf_model.predict(X_test)

print("\n--- RELATÓRIO DE DESEMPENHO DA IA ---")
print(classification_report(y_test, y_pred, target_names=['Cura (0)', 'Óbito (1)']))

# 6. Gráfico: Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Previsto Cura', 'Previsto Óbito'], 
            yticklabels=['Realmente Curou', 'Realmente Morreu'])
plt.title('Matriz de Confusão: Acertos e Erros da IA', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(f'{pasta_ml}/01_matriz_confusao.png', dpi=300)
plt.close()

# 7. Gráfico: Importância das Variáveis
importancias = rf_model.feature_importances_
df_importancia = pd.DataFrame({
    'Fator': X.columns,
    'Peso_Importancia': importancias
}).sort_values(by='Peso_Importancia', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Peso_Importancia', y='Fator', data=df_importancia, palette='rocket')
plt.title('Feature Importance: O que mais pesa para o Óbito por SRAG?', fontsize=15, pad=15)
plt.xlabel('Nível de Importância (0 a 1)')
plt.ylabel('Fatores Analisados')
plt.tight_layout()
plt.savefig(f'{pasta_ml}/02_feature_importance.png', dpi=300)
plt.close()

print(f"\nModelagem concluída! Gráficos salvos em '{pasta_ml}'.")