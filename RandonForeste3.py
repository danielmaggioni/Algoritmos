import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import numpy as np
from sklearn.inspection import permutation_importance

# Carregar os dados do arquivo Excel
caminho_arquivo = 'Sirley.xlsx'
dados = pd.read_excel(caminho_arquivo)

# Definir a variável independente (X) e a variável dependente (y)
X = dados.drop(columns=['Goodwil'])  # Variáveis independentes
y = dados['Goodwil']  # Goodwil é a variável dependente

# Dividir os dados em conjunto de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar as características
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# Definir os parâmetros para o Grid Search
parametros = {
    'n_estimators': [100, 200, 300],
    'max_depth': [50, 75, 100],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Instanciar o modelo Random Forest
modelo_rf = RandomForestRegressor(random_state=42)

# Instanciar o Grid Search com o modelo e os parâmetros definidos
grid_search = GridSearchCV(estimator=modelo_rf, param_grid=parametros, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

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

# Visualizar a importância das características
importancias_caracteristicas = melhor_modelo.feature_importances_
indices_caracteristicas = np.argsort(importancias_caracteristicas)[::-1]

print("Importância das Características:")
for i, idx in enumerate(indices_caracteristicas):
    print(f"{i+1}. {X.columns[idx]}: {importancias_caracteristicas[idx]}")

# Realizar o teste de significância com permutação
resultado_permutacao = permutation_importance(melhor_modelo, X_teste_scaled, y_teste, n_repeats=30, random_state=42)

print("\nTeste de Significância com Permutação:")
for i, idx in enumerate(indices_caracteristicas):
    print(f"{i+1}. {X.columns[idx]}: Média={resultado_permutacao.importances_mean[idx]}, Std={resultado_permutacao.importances_std[idx]}")


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

# Plotar uma das árvores do modelo Random Forest
arvore_selecionada = melhor_modelo.estimators_[0]  # Seleciona a primeira árvore do modelo

# Plotar a árvore selecionada
plt.figure(figsize=(50, 5))
plot_tree(arvore_selecionada, feature_names=X.columns, filled=True, rounded=True)
plt.show()

# Visualizar a importância das características
importancias_caracteristicas = melhor_modelo.feature_importances_
indices_caracteristicas = np.argsort(importancias_caracteristicas)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importancias_caracteristicas[indices_caracteristicas], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices_caracteristicas], rotation=90)
plt.xlabel('Característica')
plt.ylabel('Importância')
plt.title('Importância das Características')
plt.show()

# Realizar o teste de significância com permutação
resultado_permutacao = permutation_importance(melhor_modelo, X_teste_scaled, y_teste, n_repeats=30, random_state=42)

plt.figure(figsize=(10, 6))
plt.bar(range(len(X.columns)), resultado_permutacao.importances_mean[indices_caracteristicas], yerr=resultado_permutacao.importances_std[indices_caracteristicas])
plt.xticks(range(len(X.columns)), X.columns[indices_caracteristicas], rotation=90)
plt.xlabel('Característica')
plt.ylabel('Importância')
plt.title('Teste de Significância com Permutação')
plt.show()
