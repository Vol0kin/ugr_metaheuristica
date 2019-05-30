import numpy as np

try:
    from pykdtree.kdtree import KDTree                      # Implementacion paralela de KDTree
except ImportError:
    from scipy.spatial import cKDTree as KDTree             # En caso de fallar, importar cKDTree como KDTree

# Modulo para medir valores de reduccion, precision y fitness

def reduction_rate(w, threshold=0.2):
    """
    Funcion que calcula la tasa de reduccion de un vector w.
    Dice la proporcion de elementos que se encuentran por debajo de un 
    determinado umbral.

    :param w: Vector de pesos
    :param threshold: Umbral de los valores de w, se computabilizan los
                      que se encuentran por debajo (por defecto 0.2) 

    :return Devuelve la tasa de reduccion (reduction)
    """

    # Obtener el numero de elementos de w y los que estan por debajo de 
    # threshold
    N = w.shape[0]
    removed = w[w < threshold].shape[0]

    # Calcular la tasa de reduccion
    reduction = removed / N 

    return reduction

def accuracy_rate(true_Y, predicted_Y):
    """
    Funcion que calcula la tasa de aciertos.
    Estima cuantas etiquetas se han predicho de forma correcta.

    :param true_Y: Etiquetas reales
    :param predicted_y: Etiquetas predichas

    :return Devuelve el ratio de etiquetas predichas correctamente
    """
    
    # Obtener numero de etiquetas y correctamente predichas
    N = true_Y.shape[0]
    correct_pred = true_Y[true_Y == predicted_Y].shape[0]

    # Calcular tasa de aciertos
    accuracy = correct_pred / N 

    return accuracy

def fitness(accuracy, reduction, alpha=0.5):
    """
    Funcion que calcula el fitness.

    :param accuracy: Precision del clasificador (porcentaje de aciertos)
    :param reduction: Numero de caracteristicas eliminadas (w_i < 0.2)
    :param alpha: Ponderacion que se le asigna a cada parametro

    :return Devuelve valor fitness
    """

    fitness = alpha * accuracy + (1 - alpha) * reduction

    return fitness

def evaluate(X, Y, w):
    """
    Funcion para evaluar los pesos con la funcion fitness para determinar
    como de buenos son estos en la busqueda local.

    :param X: Conjunto de vectores de caracteristicas
    :param Y: Conjunto de etiquetas
    :param w: Pesos

    :return Devuelve una evaluacion de como de buenos son los nuevos pesos
            segun la funcion fitness para la tasa de aciertos y la tasa de
            reduccion
    """

    # Aplicar los pesos sobre X (aquellos con valor >= 0.2)
    weighted_X = (X * w)[:, w >= 0.2]

    # Crear un KDTree con todos los X
    kdtree = KDTree(weighted_X)

    # Para todos los elementos de X, buscar su vecino mas
    # cercano segun el criterio leave-one-out
    # Se vectoriza la operacion para ahorrar tiempo
    # Se obtienen solo los indices de los vecinos segun el criterio
    # leave-one-out
    neighbors = kdtree.query(weighted_X, k=2)[1][:, 1]

    # Predecir las etiquetas
    predicted_Y = Y[neighbors]

    # Calcular las tasas de acierto, reduccion y valor fitness
    accuracy = accuracy_rate(Y, predicted_Y)
    reduction = reduction_rate(w)
    fitness_val = fitness(accuracy, reduction)

    return fitness_val


def evaluate_population(data, labels, population):
    """
    Funcion para evaluar una poblacion entera

    :param data: Conjunto de vectores de caracteristicas
    :param labesl: Conjunto de etiquetas
    :param population: Poblacion de individuos

    :return Devuelve una lista con los valores fitness de cada elemento
    """

    # Crear lista vacia de valores fitness
    fitness_values = []

    # Evaluar cada individuo de la poblacion
    for w in population:
        fitness_values.append(evaluate(data, labels, w))

    return np.array(fitness_values)
