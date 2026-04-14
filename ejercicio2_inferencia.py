# =============================================================================
# PRÁCTICA FINAL — EJERCICIO 2
# Inferencia con Scikit-Learn
# =============================================================================

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ------------------------------------------------------------------------------------------------------
# REGRESIÓN LINEAL
# ------------------------------------------------------------------------------------------------------

# Entrena el modelo con los datos de entrenamiento.
# Evalúa sobre el test set calculando: MAE, RMSE y R².
# Genera el gráfico de residuos (valores predichos en X, residuos en Y).
# Comenta los resultados en Respuestas.md: ¿el modelo es bueno?, ¿hay overfitting o underfitting?, ¿qué variables son más influyentes?


# 1. Entrenar modelo y calcular predicciones
def entrenar_modelo(X: pd.DataFrame, y: pd.Series):

    """
    Entrena un modelo de regresión lineal.

    Devuelve la regresión (var. LinearRegression) y un diccionario con:
        * Coeficiente R^2
        * Intercepto (b_0)
        * Pesos (b_i, i > 0)
    """

    reg = LinearRegression().fit(X, y)

    reg_results = {
        'R2': reg.score(X, y),
        'Intercept': reg.intercept_,
        'Weights': reg.coef_
    }

    return reg, reg_results 

def predecir_valores(reg: LinearRegression, X: pd.DataFrame, y: pd.Series):

    """
    Predice valores para variables X usando la regresión dada reg.

    Devuelve las predicciones y los residuos.
    """

    pred =  reg.predict(X)

    e = y - pred

    return pred, e


# 2. Gráfica de residuos
def plot_residuos(pred_y: list, e: list):
    plt.figure(figsize = (8, 5))
    plt.scatter(pred_y, e, zorder = 3)
    plt.xlabel('Predicciones')
    plt.ylabel('Errores')
    plt.title('GRÁFICO DE RESIDUOS')
    plt.grid()
    plt.savefig("output/ej2_residuos.png")
    plt.show()


# 3. Extra: Q-Q plot y comparación y-hat(y)

def qq_plot(e: list):
    plt.figure(figsize=(8,5))
    stats.probplot(e, dist = 'norm', plot = plt)
    plt.title('Q-Q PLOT')
    plt.grid()
    plt.show()


def plot_y(pred_y: list, y_test: list):
    plt.figure(figsize = (8, 5))
    plt.scatter(y_test, pred_y, zorder = 3)
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title('COMPARACIÓN Y-hat(Y)')
    plt.grid()
    plt.savefig("output/ej2_residuos.png")
    plt.show()



# 3. Archivo .txt
def escribir_txt(reg_info: dict):
    
    with open("output/ej2_metricas_regresion.txt", "w", encoding="utf-8") as archivo:
        archivo.write("PARÁMETROS Y ERRORES RESULTANTES DE LA REGRESIÓN:\n\n")
        for k, v in reg_info.items():
            archivo.write(f"{k}: {v}\n")



# ------------------------------------------------------------------------------------------------------
# RESULTADO
# ------------------------------------------------------------------------------------------------------


def main():

    df = pd.read_csv("data/anthropometric_clean.csv")

    # Binarizando 'gender'
    df['gender'] = df['gender'].str.strip("'").map({'M': 0, 'F': 1})

    # Definiendo variables independientes y objetivo
    y = df['weight']
    x = df[['gender', 'age', 'height', 'arms_reach']]

    # División de datos para training-testing
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 42)

    # Entrenando modelo
    reg, reg_results = entrenar_modelo(X_train, y_train)

    # Predicción de valores
    pred_y, e = predecir_valores(reg, X_test, y_test)

    # Determinando overfitting/underfitting
    reg_results['R2 (test)'] = r2_score(y_test, pred_y)

    # Cálculo de errores MAE y MSE
    reg_results['MAE'] = mean_absolute_error(y_test, pred_y)
    reg_results['RMSE'] = np.sqrt(mean_squared_error(y_test, pred_y))

    # Guardando parámetros
    escribir_txt(reg_results)

    # Gráfica de residuos
    plot_residuos(pred_y, e)

    # Otras gráficas
    qq_plot(e)
    plot_y(pred_y, y_test)


if __name__ == '__main__':
    main()


