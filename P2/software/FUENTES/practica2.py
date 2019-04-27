import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score                  # Medir la precision de los resultados de test
from sklearn.neighbors import KNeighborsClassifier          # Clasificador KNN
import time                                                 # Medir el tiempo
import sys                                                  # Argumentos de la linea de comandos

try:
    from pykdtree.kdtree import KDTree                      # Implementacion paralela de KDTree
except ImportError:
    from scipy.spatial import cKDTree as KDTree             # En caso de fallar, importar cKDTree como KDTree

import algorithms.kfold
import algorithms.genetics
import algorithms.metrics

def genetic_classifier(train_part, test_part):
    """
    Implementacion de un clasificador 1-NN con la tecnica de la Busqueda Local
    para el calculo de los pesos.
    Ajusta un clasificador 1-NN con pesos, los cuales se obtienen mediante
    una Busqueda Local para un conjunto de entrenamiento. Despues entrena un 
    clasificador 1-NN de sklearn con los valores de entrenamiento ponderados.

    :param train_part: Particiones de entrenamiento con las que se va
                       a entrenar el clasificador 1-NN.
    :param test_part: Particiones de prueba con las que se va a comprobar
                      el ajuste del clasificador.
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
    
    cross_func = algorithms.genetics.blx_alpha_crossover

    # Para cada elemento de las listas de particiones de entrenamiento
    # y prueba, obtener los w, entrenar el modelo y predecir las etiquetas
    # Comprobar luego la precision del ajuste
    for train, test in zip(train_part, test_part):
        # Tiempo antes de lanzar la Busqueda Local
        t1 = time.time()

        # Calculo de los pesos mediante la Busqueda Local
        w = algorithms.genetics.genetic_algorithm(train[0], train[1], cross_func, generational=False)

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
        reduction = algorithms.metrics.reduction_rate(w)
        fit_val = algorithms.metrics.fitness(accuracy, reduction)
        
        # Insertar los datos en las listas
        reduction_list.append(reduction)
        accuracy_list.append(accuracy)
        aggregation_list.append(fit_val)
        time_list.append(total_time)

    return np.array(accuracy_list), np.array(reduction_list), np.array(aggregation_list), np.array(time_list)



csv_file = '../BIN/texture.csv'

# Cargar el archivo csv que contiene los datos y obtener la muestra
df = pd.read_csv(csv_file)
sample = df.values[:, 1:]

# Obtener los valores x, y de la muestra (normalizar x)
sample_x = algorithms.kfold.normalize_data(sample[:, :-1])
sample_y = sample[:, -1].reshape(-1,)

# Dividir la muestra en particiones disjuntas 
# Se obtienen 5 particiones, organizadas en 2 listas 
# La primera contiene 5 particiones de entrenamiento con sus (x, y)
# La segunda contiene 5 particiones de test con sus (x, y)
train_part, test_part = algorithms.kfold.stratify_sample(sample_x, sample_y)

np.random.seed(8912374)

class_rate, red_rate, aggrupation, times = genetic_classifier(train_part, test_part)


print('Tiempo total: {}'.format(times.sum()))

class_rate *= 100
red_rate *= 100 
aggrupation *= 100

data = np.array([[clas, rate, agr, time] for clas, rate, agr, time in zip(class_rate, red_rate, aggrupation, times)])

# Crear un DataFrame con los datos de salida (particiones)
out_df = pd.DataFrame(data, columns=['%_clas', '%_red', 'Agr.', 'T'], index=['Particion {}'.format(i) for i in range(1, 6)])

# Mostrar datos de las particiones
print('\nResultados de las ejecuciones\n')
print(out_df)

# Mostrar valores estadisticos por pantalla
print('\nValores estadisticos\n')
stat_data = np.array([[class_rate.max(), red_rate.max(), aggrupation.max(), times.max()],
                      [class_rate.min(), red_rate.min(), aggrupation.min(), times.min()],
                      [class_rate.mean(), red_rate.mean(), aggrupation.mean(), times.mean()],
                      [np.median(class_rate), np.median(red_rate), np.median(aggrupation), np.median(times)],
                      [np.std(class_rate), np.std(red_rate), np.std(aggrupation), np.std(times)]])

stat_df = pd.DataFrame(stat_data, index=['Maximo', 'Minimo', 'Media', 'Mediana', 'Desv. tipica'],
                       columns=['%_clas', '%_red', 'Agr.', 'T'])

print(stat_df)
