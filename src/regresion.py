import math
import random
import matplotlib.pyplot as plt
import numpy as np




# Parameters:
a = 2.0
b = 1.0
xmin = 0.0
xmax = 10.0
noise = 2.0
n = 100

# Randomly generated problem data:
np.random.seed(42)
x = xmin + np.random.rand(n)*(xmax - xmin)
t = a*x + b + np.random.randn(n)*noise

    
def regresion_lineal_simple(x, t, num_iters=8, eta=0.01):
        w = np.random.randn()
        b = np.random.randn()

        xmin, xmax = x.min(), x.max()

        n = len(x)

        for i in range(num_iters):

            # Definición de la recta
            y = b + w * x 

            #Cálculo del error
            e = y - t
            mse = sum(t - y)**2 / n

            #Cálculo del gradiente de los pesos b, w1
            db = 2 * sum(y - t) / n
            dg = 2 * sum(y - t) * x / n

            #Actualización de parametros
            b = b - eta * db
            w = w - eta * dg
            

            plt.plot(x, t, 'o', label='Datos reales')
            plt.plot([xmin, xmax], [w * xmin + b, w * xmax + b], 'grey', label=f'Modelo (ite {i + 1})')
        plt.plot([xmin, xmax], [w * xmin + b, w * xmax + b], 'r-', label='Modelo final')
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("t")
        plt.show()

        return w, b

def show_data():

    plt.figure(figsize=(6, 6))
    plt.plot(x, t, 'o')
    plt.plot([xmin, xmax], [a*xmin + b, a*xmax + b], 'r-')
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

def main():
    regresion_lineal_simple(x = x, t = t)
    show_data()


if "__name__" == "__main__":
    main()