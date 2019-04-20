import numpy as np

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
