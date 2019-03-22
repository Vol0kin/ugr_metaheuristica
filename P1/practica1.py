import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold         # Particionar muestra
from sklearn.metrics import accuracy_score                  # Medir la precision de los resultados de test
from sklearn.neighbors import KNeighborsClassifier          # Clasificador KNN
from pykdtree.kdtree import KDTree                          # Implementacion paralela de KDTree 
import timeit

def normalize_data(sample):
    """
    Funcion para normalizar los datos de una muestra. Convierte
    las caracteristicas al rango [0, 1].

    :param sample: Muestra a normalizar.

    :return new_sample: Nueva muestra normalizada en el intervalo
                        [0, 1]
    """
    # Obtener minimo y maximo de cada columna de la muestra
    # y la diferencia entre estos
    sample_min = sample.min(axis=0)
    sample_max = sample.max(axis=0)
    sample_diff = sample_max - sample_min

    # Normalizar los datos 
    new_sample = (sample - sample_min) / sample_diff

    # Si algun valor es nan, se cambia por 0
    new_sample = np.nan_to_num(new_sample)

    return new_sample

def stratify_sample(X, y):
    """
    Funcion para dividir una muestra en particiones disjuntas 
    que conservan la proporcion de las clases.
    Divide la muestra en 5 particiones disjuntas y las devuelve.

    :param X: Caracteristicas de la muestra.
    :param y: Etiquetas de la muestra.

    :return Devuelve dos listas: train_part, que contiene listas 
            formadas por [X_train, y_train] (una sublista por cada 
            una de las 5 particiones) y test_part, que contiene listas 
            formadas por [X_test, y_test] (una sublista por cada una 
            de las 5 particiones)
    """
    # Crear un nuevo estratificador que divida la muestra en 
    # 5 partes disjuntas (proporcion 80-20)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

    # Listas para los datos de entrenamiento y test
    train_part = []
    test_part = []

    # Para cada particion, obtener los indices de entrenamiento y test 
    # Crear X e y de entrenamiento y test 
    # Meterlos en las listas correspondientes
    for train_index, test_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        train_part.append([X_train, y_train])
        test_part.append([X_test, y_test])

    return train_part, test_part

def KNN(X, y, x, neighbors=1):
    """
    Implementacion del KNN vectorizada.

    Recibe una muestra X y sus etiquetas, y un array
    x del que buscar sus vecinos mas cercanos. Construye
    un kdtree con X y busca los k vecinos mas cercanos de
    x.

    :param X: Vectores de caracteristicas de una muestra de N elementos
    :param y: Etiquetas que se corresponden a cada uno de los N elementos de X
    :param x: Elemento o array de elementos de los que buscar sus k-vecinos mas cercanos
    :param neighbors: Numero de vecinos que buscar (por defecto 1)

    :return: K vecions mas cercanos de x
    """
    # Construir el arbol
    kd_tree = KDTree(X)

    # Obtener las distancias y los indices de los k vecions mas cercanos
    dist, index = kd_tree.query(x, k=neighbors)

    return y[index]


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

def knn_classifier(train_part, test_part):
    """
    Implementacion del clasificador 1-NN normal para la clasificacion 
    de elementos.
    Entrena un modelo de sklearn de K vecinos mas cercanos para poder
    predecir el vecino mas cercano. Despues comprueba como de bueno es
    el ajuste.

    :param train_part: Particiones de entrenamiento con las que se va
                       a entrenar el clasificador 1-NN.
    :param test_part: Particiones de prueba con las que se va a comprobar
                      el ajuste del clasificador.
    """
    # Crear un nuevo clasificador de sklearn del tipo 1-NN
    neigh = KNeighborsClassifier(n_neighbors=1)

    # Establecer que la tasa de reduccion es 0 (no se elimina
    # ninguna caracteristica)
    reduction = 0

    # Para cada elemento de las listas de particiones de entrenamiento
    # y prueba, entrenar el modelo y predecir las etiquetas
    # Comprobar luego la precision del ajuste
    for train, test in zip(train_part, test_part):
        # Entrenar el modelo
        neigh.fit(train[0], train[1])

        # Predecir etiquetas
        knn_labels = neigh.predict(test[0])

        # Calcular precision de la prediccion y valor fitness
        accuracy = accuracy_score(test[1], knn_labels)
        print("Accuracy: {}".format(accuracy))
        fit_val = fitness(accuracy, reduction)
        print(fit_val)


def relief(X, y):
    N = y.shape[0]
    w = np.zeros((N,), np.float64)

    for x in X:
        print("hola")


df = pd.read_csv('data/colposcopy.csv')
sample = df.values[:, 1:]

# Obtener los valores x, y de la muestra (normalizar x)
sample_x = normalize_data(sample[:, :-1])
sample_y = sample[:, -1].reshape(-1,)

# Dividir la muestra en particiones disjuntas 
# Se obtienen 5 particiones, organizadas en 2 listas 
# La primera contiene 5 particiones de entrenamiento con sus (x, y)
# La segunda contiene 5 particiones de test con sus (x, y)
train_part, test_part = stratify_sample(sample_x, sample_y)

knn_classifier(train_part, test_part)
