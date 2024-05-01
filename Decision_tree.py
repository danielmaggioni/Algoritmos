import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.7, random_state=42)

# Escalar as características
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# Definir os parâmetros para o Grid Search
parametros = {
    'max_depth': [50, 75, 100],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Instanciar o modelo Árvore de Decisão
modelo_arvore = DecisionTreeRegressor(random_state=42)

# Instanciar o Grid Search com o modelo e os parâmetros definidos
grid_search = GridSearchCV(estimator=modelo_arvore, param_grid=parametros, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Treinar o Grid Search com os dados de treinamento
grid_search.fit(X_treino_scaled, y_treino)

# Obter os melhores parâmetros encontrados pelo Grid Search
melhores_parametros = grid_search.best_params_
print("Melhores Parâmetros Encontrados:", melhores_parametros)

# Fazer previsões nos dados de teste usando o melhor modelo encontrado
melhor_modelo = grid_search.best_estimator_
previsoes = melhor_modelo.predict(X_teste_scaled)

# Avaliar o modelo
mse = mean_squared_error(y_teste, previsoes)
rmse = mean_squared_error(y_teste, previsoes, squared=False)
mae = mean_absolute_error(y_teste, previsoes)  # Adicionando o cálculo do MAE
r2 = r2_score(y_teste, previsoes)

print("Erro Quadrático Médio (MSE):", mse)
print("Raiz do Erro Quadrático Médio (RMSE):", rmse)
print("Erro Absoluto Médio (MAE):", mae)  # Imprimindo o valor do MAE
print("Coeficiente de Determinação (R²):", r2)

# Plotar o valor real versus o valor previsto
plt.scatter(y_teste, previsoes, color='b', label='Valores Previstos')
plt.plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], 'g--', lw=4, label='Linha de Referência')
plt.xlabel('Valor Real')
plt.ylabel('Valor Previsto')
plt.title('Valor Real vs Valor Previsto')
plt.legend()
plt.show()

# Plotar mapa de calor
correlacoes = dados.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlacoes, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor das Correlações')
plt.show()
