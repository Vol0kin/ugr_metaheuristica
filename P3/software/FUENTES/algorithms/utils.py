import numpy as np

# Funciones utiles que se pueden utilizar en los algoritmos

def normalize_w(w):
    """
    Funcion para normalizar los pesos w en el rango [0, 1].
    Si algun wi no esta en ese rango, se le asigna el valor mas cercano
    dependiendo de por que extremo se pase (si es menor que 0 se le asigna
    el 0, si es mayor que 1 se le asigna 1).

    :param w: Vector de pesos a normalizar

    :return Devuelve normal_w, donde se han normalizado los pesos.
    """

    # Copiar w 
    normal_w = np.copy(w)

    normal_w[normal_w < 0.0] = 0.0
    normal_w[normal_w > 1.0] = 1.0

    return normal_w

def generate_initial_w(N):
    """
    Funcion para generar un valor de w inicial con valores aleatorios
    uniformes en el rango [0, 1]

    :param Numero de valores aletorios que generar

    :return Devuelve un nuevo vector w con valores aleatorios
    """

    return np.random.uniform(0.0, 1.0, N)
