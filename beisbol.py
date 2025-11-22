import pandas as pd # Librería para manejar los datos
import numpy as np # Librería para los calculos numéricos
import matplotlib.pyplot as plt # Librería para graficar
import seaborn as sns # Libreria para realizar las gráficas con estilo 
from sklearn.model_selection import train_test_split # Libreria de Sklearn para dividir los datos en test y tranin
from sklearn.linear_model import LinearRegression # Modelo de Regresión Lineal
from sklearn.model_selection import GridSearchCV # Librería para la busqueda de hiperparpárametros
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Métricas para evaluar el modelo
from sklearn.preprocessing import StandardScaler # Librería para escalar los datos

datos = pd.read_csv("data/beisbol.csv") # Cargamos los datos

x = datos[['bateos']] # Variable independiente
y = datos['runs'] # Variable dependiente

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,) # Dividimos las datos de entrenamiento y prueba
print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")

modelo = LinearRegression() # Creamos el modelo de regresióin lineal
modelo.fit(X_train, y_train) # Entrenamos el modelo
result = modelo.score(X_train, y_train) # Evaluamos el modelo con los datos de prueba

print("Score del 1er modelo: ", result)

y_pred = modelo.predict(X_test) # Realizamos la predicción del modelo

mae = mean_absolute_error(y_test, y_pred) # Calculamos el MAE (Error Absoluto Medio)
mse = mean_squared_error(y_test, y_pred) # Calculamos el MSE (Error Cuadrático Medio)
rmse = np.sqrt(mse) # Calculamos el RMSE (Raíz del Error Cuadrático Medio)
r2 = r2_score(y_test, y_pred) # Cualculamos R cuadrada

print("\nMétricas - 1er Modelo")
print(f"MAE (Error Absoluto Medio):  {mae:.4f}")
print(f"MSE (Error Cuadrático Medio):  {mse:.4f}")
print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.4f}")
print(f"R²: {r2:.4f}")


scaler = StandardScaler() # Comenzamos con la optimización del modelo

X_train_scaled = scaler.fit_transform(X_train) # Escalamos los datos de entrenamiento
X_test_scaled  = scaler.transform(X_test) # Escalamos los datos de prueba

modelo2 = LinearRegression() # Creamos un nuevo modelo de regresión lineal
modelo2.fit(X_train_scaled, y_train) # Entrenamos el nuevo modelo con los datos escalados

result2 = modelo2.score(X_train_scaled, y_train) # Evaluamos el nuevo score del modelo de datos
print("Score del 2do modelo: ", result2)

y_pred2 = modelo2.predict(X_test_scaled) # Realizamos la predicción del nuevo modelo

mae2 = mean_absolute_error(y_test, y_pred2) # Calculamos el MAE del nuevo modelo
rmse2 = np.sqrt(mean_squared_error(y_test, y_pred2)) # Calculamos el RMSE dek nuevo modelo
r22 = r2_score(y_test, y_pred2) # Calculamos R cuadrada del nuevo modelo

print("\nMétricas - 2do Modelo")
print(f"MAE (Error Absoluto Medio):  {mae2:.4f}")
print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse2:.4f}")
print(f"R²: {r22:.4f}")

print("\nOptimización Básica del Modelo - Escalado de Datos")
print("Modelo sin escalar:") 
print(f"RMSE = {rmse:.4f} | R² = {r2:.4f}") # Resultados del primer modelo

print("\nModelo con escalado:")
print(f"RMSE = {rmse2:.4f} | R² = {r22:.4f}") # Resultados de segundo modelo ya optimizado

mejor = "2do Modelo" if r22 > r2 else "1er Modelo"
print(f"\nEl mejor modelo es: {mejor}") # Comparación de ambos modelos y el mejor

print("\nHiperparámetros")
# Definimos los hiperparámetros que queremos probar en el modelo LinearRegression
parametros = {
    "fit_intercept": [True, False], # Probar con y sin término independiente
    "positive": [True, False] # Probar si forzamos coeficientes positivos o no
}

# Configuramos GridSearchCV para buscar la mejor combinación de hiperparámetros
grid = GridSearchCV(
    estimator=LinearRegression(), # Modelo base
    param_grid=parametros, # Diccionario con hiperparámetros a evaluar
    scoring='neg_mean_squared_error', # Métrica para evaluar cada combinación
    cv=5 # Validación cruzada con 5 particiones
)

grid.fit(X_train_scaled, y_train) # Entrenamos la búsqueda usando los datos escalados
print("\nMejores parámetros encontrados:", grid.best_params_)


# Gráfica — Dispersión de datos
plt.figure(figsize=(7,5)) # Tamaño de la figura de la gráfica
sns.scatterplot(
x=datos['bateos'], 
y=datos['runs'],
color='blue',      # Color para los puntos
edgecolor='black'  # Contorno
)
plt.title("Dispersión: Bateos vs Runs", color='darkred')
plt.xlabel("Bateos", color='navy')
plt.ylabel("Runs", color='green')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show() # Mostramos la gráfica

# Gráfica — Reales vs Predichos
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', label="Línea Ideal")
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.title("Comparación entre valores reales y predichos")
plt.legend()
plt.show()

