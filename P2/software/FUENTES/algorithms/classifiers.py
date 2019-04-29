import numpy as np
import time																									# Medir tiempos
from enum import Enum																				# Enumerados
from sklearn.metrics import accuracy_score                  # Medir la precision de los resultados de test
from sklearn.neighbors import KNeighborsClassifier          # Clasificador KNN
from . import genetics
from . import metrics
from . import memetics
from . import kfold

class GeneticCross(Enum):
		BLX = 1
		AC = 2

class GeneticReplacement(Enum):
		GENERATIONAL = 1
		STATIONARY = 2

class MemeticsLocalSearch(Enum):
		ALL_POPULATION = 1
		BEST_CHROMOSOMES = 2
		RANDOM_CHROMOSOMES = 3

def genetic_classifier(train_part, test_part, cross, replacement):
    """
    Implementacion de un clasificador 1-NN con la tecnica de los Algoritmos
    Geneticos para el calculo de los pesos.
    Ajusta un clasificador 1-NN con pesos, los cuales se obtienen mediante
    Algoritmos Geneticos para un conjunto de entrenamiento mediante distintos
    criterios, como por ejemplo con distintos operadores de cruce o con
    distintos criterios de reemplazamiento.
    Despues entrena un clasificador 1-NN de sklearn con los valores de
    entrenamiento ponderados.

    :param train_part: Particiones de entrenamiento con las que se va
                       a entrenar el clasificador 1-NN.
    :param test_part: Particiones de prueba con las que se va a comprobar
                      el ajuste del clasificador.
    :param cross: Tipo de cruce (BLX-alfa o cruce aritmetico)
    :param replacement: Tipo de reemplazo (generacional o estacionario)
    
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
    
    # Determinar el operador de cruce a utilizar
    if cross == GeneticCross.BLX:
    		cross_func = genetics.blx_alpha_crossover
    else:
    		cross_func = genetics.arithmetic_crossover
    
    # Determinar la estrategia de reemplazo a utilizar
    if replacement == GeneticReplacement.GENERATIONAL:
    		generational = True
    else:
    		generational = False

    # Para cada elemento de las listas de particiones de entrenamiento
    # y prueba, obtener los w, entrenar el modelo y predecir las etiquetas
    # Comprobar luego la precision del ajuste
    for train, test in zip(train_part, test_part):
        # Tiempo antes de lanzar la Busqueda Local
        t1 = time.time()

        # Calculo de los pesos mediante el Algoritmo Genetico
        w = genetics.genetic_algorithm(train[0], train[1], cross_func, generational)

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


def memetic_classifier(train_part, test_part, ls_evaluations):
    """
    Implementacion de un clasificador 1-NN con la tecnica de los Algoritmos
    Memeticos para el calculo de los pesos.
    Ajusta un clasificador 1-NN con pesos, los cuales se obtienen mediante
    Algoritmos Memeticos para un conjunto de entrenamiento mediante distintos
    criterios, como por ejemplo la cantidad de cromosomas a evaluar en la busqueda
    local y si estos tienen que ser los mejores o aleatorios.
    Despues entrena un clasificador 1-NN de sklearn con los valores de
    entrenamiento ponderados.

    :param train_part: Particiones de entrenamiento con las que se va
                       a entrenar el clasificador 1-NN.
    :param test_part: Particiones de prueba con las que se va a comprobar
                      el ajuste del clasificador.
    :param ls_evaluations: Tipo de evaluaciones que se haran en la LS
    
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
    
    # Decidir los ratios del algoritmo memetico y si se hace con los mejores
    if ls_evaluations == MemeticsLocalSearch.ALL_POPULATION:
    		ls_rate = 1.0
    		ls_best = False
    elif ls_evaluations == MemeticsLocalSearch.BEST_CHROMOSOMES:
    		ls_rate = 0.1
    		ls_best = True
    else:
    		ls_rate = 0.1
    		ls_best = False

    # Para cada elemento de las listas de particiones de entrenamiento
    # y prueba, obtener los w, entrenar el modelo y predecir las etiquetas
    # Comprobar luego la precision del ajuste
    for train, test in zip(train_part, test_part):
        # Tiempo antes de lanzar la Busqueda Local
        t1 = time.time()

        # Calculo de los pesos mediante la Busqueda Local
        w = memetics.memetic_algorithm(train[0], train[1], ls_rate, ls_best=ls_best)

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


