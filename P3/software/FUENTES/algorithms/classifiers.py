import numpy as np
import time																									# Medir tiempos
from enum import Enum																				# Enumerados
from sklearn.metrics import accuracy_score                  # Medir la precision de los resultados de test
from sklearn.neighbors import KNeighborsClassifier          # Clasificador KNN
from . import metrics
from . import kfold
from . import ils
from . import simulated_annealing as sa

# Modulo para los distintos clasificadores implementados

class SearchAlgorithm(Enum):
    SA = 1
    ILS = 2


def classifier(train_part, test_part, search_algorithm):
    """
    Implementacion de un clasificador 1-NN con distintas tecnicas
    metaheuristicas para aprender el peso de las caracteristicas.
    Ajusta un clasificador 1-NN con pesos, los cuales se obtienen mediante
    algoritmos de busqueda basados en trayectorias y evolucion diferencial.
    Despues entrena un clasificador 1-NN de sklearn con los valores de
    entrenamiento ponderados.

    :param train_part: Particiones de entrenamiento con las que se va
                       a entrenar el clasificador 1-NN.
    :param test_part: Particiones de prueba con las que se va a comprobar
                      el ajuste del clasificador.
    :param search_algorithm: Algoritmo de busqueda. Puede ser:
                                - SA (Simulated Annealing)
                                - ILS (Iterative Local Search)
                                - DE (Differential Evolution)
                                - DER (Differential Evolution Random cross)
    
    :return Devuelve arrays creados a apartir de las listas de tasas de 
            clasificacion, tasas de reduccion, agrupaciones y tiempos.
    """

    # Crear un nuevo clasificador 1-NN de sklearn
    neigh = KNeighborsClassifier(n_neighbors=1)

    # Listas para guardar los datos (reduccion, tasa de clas. y tiempo)
    reduction_list = []
    accuracy_list = []
    aggregation_list = []
    time_list = []

    # Para cada elemento de las listas de particiones de entrenamiento
    # y prueba, obtener los w, entrenar el modelo y predecir las etiquetas
    # Comprobar luego la precision del ajuste
    for train, test in zip(train_part, test_part):
        # Determinar funcion a ejecutar
        if search_algorithm == SearchAlgorithm.SA:
            weights_func = sa.simulated_annealing
        elif search_algorithm == SearchAlgorithm.ILS:
            weights_func = ils.ils

        # Establecer parametros de los algoritmos
        params = {'X': train[0], 'y': train[1]}

        # Tiempo antes de lanzar la Busqueda Local
        t1 = time.time()

        # Calculo de los pesos mediante el Algoritmo Genetico
        w = weights_func(**params)

        # Tiempo despues de terminar la Busqueda Local
        t2 = time.time()
        total_time = t2 - t1

        # Calcular los X de entrenamiento aplicando los pesos y entrenar
        # el modelo con estos valores
        # Se seleccionan solo aquellas caracteristicas cuyo w_i sea superior
        # o igual a 0.2
        weighted_X_train = (train[0] * w)[:, w >= 0.2]
        neigh.fit(weighted_X_train, train[1])

        # Calcular los X de test aplicando los pesos y obtener las etiquetas
        # Se seleccionan solo aquellas caracteristicas cuyo w_i sea superior
        # o igual a 0.2
        weighted_X_test = (test[0] * w)[:, w >= 0.2]
        knn_labels = neigh.predict(weighted_X_test)

        # Obtener tasa de aciertos, reduccion y agrupacion de ambos
        accuracy = accuracy_score(test[1], knn_labels)
        reduction = metrics.reduction_rate(w)
        fit_val = metrics.fitness(accuracy, reduction)
        
        # Insertar los datos en las listas
        reduction_list.append(reduction)
        accuracy_list.append(accuracy)
        aggregation_list.append(fit_val)
        time_list.append(total_time)

    return np.array(accuracy_list), np.array(reduction_list), np.array(aggregation_list), np.array(time_list)


