import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold         # Particionar muestra
from sklearn.metrics import accuracy_score                  # Medir la precision de los resultados de test
from sklearn.neighbors import KNeighborsClassifier          # Clasificador KNN
from pykdtree.kdtree import KDTree                          # Implementacion paralela de KDTree 
import time                                                 # Medir el tiempo

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

def KNN(X, Y, x, neighbors=1):
    """
    Implementacion del KNN vectorizada.

    Recibe una muestra X y sus etiquetas, y un array
    x del que buscar sus vecinos mas cercanos. Construye
    un kdtree con X y busca los k vecinos mas cercanos de
    x.

    :param X: Vectores de caracteristicas de una muestra de N elementos
    :param Y: Etiquetas que se corresponden a cada uno de los N elementos de X
    :param x: Elemento o array de elementos de los que buscar sus k-vecinos mas cercanos
    :param neighbors: Numero de vecinos que buscar (por defecto 1)

    :return: K vecions mas cercanos de x
    """

    # Construir el arbol
    kd_tree = KDTree(X)

    # Obtener las distancias y los indices de los k vecions mas cercanos
    dist, index = kd_tree.query(x, k=neighbors)

    return Y[index]

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

def relief(X, Y):
    """
    Funcion para calcular los pesos mediante el algoritmo RELIEF.

    :param X: Conjunto de vectores de caracteristicas.
    :param Y: Conjunto de etiquetas.

    :return Devuelve los pesos w que ponderan las caracteristicas.
    """

    # Obtener numero de elementos de la muestra
    N = Y.shape[0]

    # Inicializar w a [0, 0, ... , 0]
    w = np.zeros((X.shape[1],), np.float64)

    # Recorrer todos los elementos de la muestra 
    # Para cada elemento, obtener su amigo mas cercano y su enemigo 
    # mas cercano
    # Actualizar w con la nueva informacion
    for i in range(N):
        # Obtener el elemento y su etiqueta
        x, y = X[i], Y[i]

        # Obtener la lista de amigos y enemigos
        allies = X[Y == y]
        enemies = X[Y != y]

        # Construir los arboles de amigos y enemigos
        kdtree_allies = KDTree(allies)
        kdtree_enemies = KDTree(enemies)

        # Obtener el aliado mas cercano mediante la tecnica de leave-one-out 
        # Se escoge el segundo vecino mas cercano a x, ya que el primero es 
        # el mismo (un indice al segundo vecino mas cercano)
        closest_ally = kdtree_allies.query(x.reshape(1, -1), k=2)[1][0, 1]

        # Obtener el enemigo mas cercano a x (se obtiene un indice)
        closest_enemy = kdtree_enemies.query(x.reshape(1, -1), k=1)[1][0]

        # Obtener el aliado y el enemigo correspondientes
        ally = allies[closest_ally]
        enemy = enemies[closest_enemy]

        # Actualizar w
        w += abs(x - enemy) - abs(x - ally)

    # Obtener el maximo de w
    w_max = w.max()

    # Hacer que los elementos inferiores a 0 tengan como valor 0
    w[w < 0.0] = 0.0

    # Normalizar w con el maximo
    w = w / w_max

    return w


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
    reduction = 0.0

    # Para cada elemento de las listas de particiones de entrenamiento
    # y prueba, entrenar el modelo y predecir las etiquetas
    # Comprobar luego la precision del ajuste
    for train, test in zip(train_part, test_part):
        #Obtener el tiempo antes de entrenar el modelo
        t1 = time.time()

        # Entrenar el modelo
        neigh.fit(train[0], train[1])

        # Obtener el tiempo despues de entrenar el modelo
        t2 = time.time()
        total_time = t2 - t1

        # Predecir etiquetas
        knn_labels = neigh.predict(test[0])

        # Calcular precision de la prediccion y valor fitness
        accuracy = accuracy_score(test[1], knn_labels)
        fit_val = fitness(accuracy, reduction)

        print("Accuracy: {}\tReduction: {}\tAgrupacion: {}\tTiempo: {}".format(accuracy, reduction, fit_val, total_time))


def relief_classifier(train_part, test_part):
    """
    Implementacion de un clasificador 1-NN con la tecnica de RELIEF para 
    el calculo de los pesos.
    Ajusta un clasificador 1-NN con pesos, los cuales se obtienen mediante 
    la tecnica RELIEF para un conjunto de entrenamiento. Despues entrena un 
    clasificador 1-NN de sklearn con los valores de entrenamiento ponderados.

    :param train_part: Particiones de entrenamiento con las que se va
                       a entrenar el clasificador 1-NN.
    :param test_part: Particiones de prueba con las que se va a comprobar
                      el ajuste del clasificador.
    """

    # Crear un nuevo clasificador 1-NN de sklearn
    neigh = KNeighborsClassifier(n_neighbors=1)

    # Para cada elemento de las listas de particiones de entrenamiento
    # y prueba, obtener los w, entrenar el modelo y predecir las etiquetas
    # Comprobar luego la precision del ajuste
    for train, test in zip(train_part, test_part):
        # Tiempo antes de lanzar el algoritmo de RELIEF
        t1 = time.time()

        # Calculo de los pesos mediante RELIEF
        w = relief(train[0], train[1])

        # Tiempo despues de terminar RELIEF
        t2 = time.time()
        total_time = t2 - t1

        # Calcular los X de entrenamiento aplicando los pesos y entrenar
        # el modelo con estos valores
        weighted_X_train = train[0] * w
        neigh.fit(weighted_X_train, train[1])

        # Calcular los X de test aplicando los pesos y obtener las etiquetas
        weighted_X_test = test[0] * w
        knn_labels = neigh.predict(weighted_X_test)

        # Obtener tasa de aciertos, reduccion y agrupacion de ambos
        accuracy = accuracy_score(test[1], knn_labels)
        reduction = reduction_rate(w)
        fit_val = fitness(accuracy, reduction)
        
        print("Accuracy: {}\tReduction: {}\tAgrupacion: {}\tTiempo: {}".format(accuracy, reduction, fit_val, total_time))



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

relief_classifier(train_part, test_part)
