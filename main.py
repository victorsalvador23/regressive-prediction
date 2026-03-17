import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


df = pd.read_csv("train.csv")
df = df[['LotArea','OverallQual','YearBuilt','SalePrice']]

# verificar valores nulos
print(df.isnull().sum())

# preencher valores ausentes
df['LotArea'].fillna(df['LotArea'].mean(), inplace=True)
df['OverallQual'].fillna(df['OverallQual'].mean(), inplace=True)
df['YearBuilt'].fillna(df['YearBuilt'].mean(), inplace=True)

# definir X e y
X = df[['LotArea','OverallQual','YearBuilt']]
y = df['SalePrice']

# dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# modelo regressão linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# previsões
y_pred = modelo.predict(X_test)

# métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2:", r2)

# gráfico
plt.scatter(y_test, y_pred)
plt.xlabel("Valores Reais")
plt.ylabel("Valores Previstos")
plt.title("Valores Reais vs Previstos")
plt.show()