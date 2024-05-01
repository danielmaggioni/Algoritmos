import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns  

# Carregar os dados do arquivo Excel
caminho_arquivo = 'Sirley.xlsx'
dados = pd.read_excel(caminho_arquivo)

# Definir a variável independente (X) e a variável dependente (y)
X = dados.drop(columns=['Goodwil'])  # Variáveis independentes
y = dados['Goodwil']  # Goodwil é a variável dependente

# Dividir os dados em conjunto de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.1, random_state=42)

# Escalar as características
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# Instanciar o modelo de Regressão Linear
modelo_lr = LinearRegression()

# Treinar o modelo de Regressão Linear com os dados de treinamento escalados
modelo_lr.fit(X_treino_scaled, y_treino)

# Fazer previsões nos dados de teste
previsoes = modelo_lr.predict(X_teste_scaled)

# Avaliar o modelo
mse = mean_squared_error(y_teste, previsoes)
rmse = mean_squared_error(y_teste, previsoes, squared=False)
r2 = r2_score(y_teste, previsoes)
print("Erro Quadrático Médio (MSE):", mse)
print("Raiz do Erro Quadrático Médio (RMSE):", rmse)
print("Coeficiente de Determinação (R²):", r2)

# Plotar todos os valores
plt.scatter(y_teste, previsoes, color='b', label='Valores Previstos', alpha=0.5, edgecolors='none')  
plt.scatter(y_teste, y_teste, color='r', label='Valores Reais', alpha=0.5, edgecolors='none')
plt.xlabel('Valores Reais e Previstos')
plt.ylabel('Valores Reais e Previstos')
plt.title('Dispersão: Valores Reais vs Valores Previstos (Regressão Linear)')
plt.legend()
plt.show()

# Plotar mapa de calor
correlacoes = dados.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlacoes, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor das Correlações')
plt.show()
