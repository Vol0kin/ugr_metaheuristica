import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold         # Particionar muestra
from sklearn.metrics import accuracy_score                  # Medir la precision de los resultados de test
from sklearn.neighbors import KNeighborsClassifier          # Clasificador KNN
from pykdtree.kdtree import KDTree                          # Implementacion paralela de KDTree 
import time                                                 # Medir el tiempo
import sys                                                  # Argumentos de la linea de comandos

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

def local_search(X, Y, max_evaluations=15000, max_traits_evaluations=20, mean=0.0, sigma=0.3):
    """
    Funcion para el calculo de w mediante la búsqueda local.

    :param X: Conjunto de vectores de caracteristicas 
    :param Y: Conjunto de etiquetas 
    :param max_evaluations: Maximo numero de evaluaciones de la funcion fitness 
                            que puede hacer la busqueda 
    :param max_traits_evaluations: Numero maximo de veces que se pueden evaluar 
                                   las caracteristicas sin exito
    :param mean: Media del operador de mutacion 
    :param sigma: Desviacion tipica del operador de mutacion 

    :return Devuelve los w calculados
    """

    # Establecer la semilla aleatoria
    np.random.seed(8912374)

    # Obtener numero de caracteristicas
    N = X.shape[1]

    # Generar un w inicial y establecer las evaluaciones, las 
    # evaluaciones fallidas y el valor fitness iniciales
    w = np.random.uniform(0.0, 1.0, N)
    evaluations = 0
    unsuccessful_evaluations = 0
    fitness_val = evaluate(X, Y, w)

    # Mientras no se hayan superado el numero maximo de evaluaciones
    # intentar ajustar w
    while evaluations < max_evaluations:
        # Copiar w
        current_w = np.copy(w)

        # Intentar mutar cada caracteristica en el orden dado por la permutacion
        # hasta encontrar la primera mutacion que obtiene un mejor valor de fitness
        for trait in np.random.permutation(N):
            # Mutar el w_i con un valor de la normal con media mean y d.t. sigma
            w[trait] += np.random.normal(mean, sigma)

            # Normalizar los w en el rango [0, 1]
            w = normalize_w(w)
            
            # Evaluar el nuevo w con los datos
            evaluations += 1
            new_fitness = evaluate(X, Y, w)

            # Si el nuevo valor de fitness es mejor, conservar w, guardar el mejor
            # valor de fitness e indicar que se ha realizado una evaluacion con 
            # exito
            if new_fitness > fitness_val:
                fitness_val = new_fitness
                unsuccessful_evaluations = 0
                break
            # En caso contrario, contabilizar un error mas y restaurar los w
            else:
                unsuccessful_evaluations += 1
                w[trait] = current_w[trait]

            # Si se sobrepasa el numero max de evaluaciones o se obtienen demasiados 
            # errores, se termina la busqueda
            if evaluations > max_evaluations or unsuccessful_evaluations > max_traits_evaluations * N:
                return w

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
    :return Devuelve arrays creados a apartir de las listas de tasas de 
            clasificacion, tasas de reduccion, agrupaciones y tiempos.
    """

    # Crear un nuevo clasificador de sklearn del tipo 1-NN
    neigh = KNeighborsClassifier(n_neighbors=1)

    # Establecer que la tasa de reduccion es 0 (no se elimina
    # ninguna caracteristica)
    reduction = 0.0

    # Listas para guardar los datos (reduccion, tasa de clas. y tiempo)
    reduction_list = []
    accuracy_list = []
    aggregation_list = []
    time_list = []

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

        # Insertar los datos en las listas
        reduction_list.append(reduction)
        accuracy_list.append(accuracy)
        aggregation_list.append(fit_val)
        time_list.append(total_time)

    return np.array(accuracy_list), np.array(reduction_list), np.array(aggregation_list), np.array(time_list)


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
        
        # Insertar los datos en las listas
        reduction_list.append(reduction)
        accuracy_list.append(accuracy)
        aggregation_list.append(fit_val)
        time_list.append(total_time)

    return np.array(accuracy_list), np.array(reduction_list), np.array(aggregation_list), np.array(time_list)


def local_search_classifier(train_part, test_part):
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

    # Para cada elemento de las listas de particiones de entrenamiento
    # y prueba, obtener los w, entrenar el modelo y predecir las etiquetas
    # Comprobar luego la precision del ajuste
    for train, test in zip(train_part, test_part):
        # Tiempo antes de lanzar la Busqueda Local
        t1 = time.time()

        # Calculo de los pesos mediante la Busqueda Local
        w = local_search(train[0], train[1])

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
        reduction = reduction_rate(w)
        fit_val = fitness(accuracy, reduction)
        
        # Insertar los datos en las listas
        reduction_list.append(reduction)
        accuracy_list.append(accuracy)
        aggregation_list.append(fit_val)
        time_list.append(total_time)

    return np.array(accuracy_list), np.array(reduction_list), np.array(aggregation_list), np.array(time_list)

if __name__ == '__main__':
    # Posibles archivos y funciones
    files = ('colposcopy', 'ionosphere', 'texture')
    functions = ('knn', 'relief', 'local')

    # Leer archivo y funcion de entrada
    in_file = sys.argv[1]
    in_func = sys.argv[2]

    # Comprobar si el archivo no es correcto, en cuyo caso se lanza una excepcion
    if not in_file in files:
        raise ValueError('Error: el archivo tiene que ser uno de los siguitenes: {}'.format(files))

    # Comprobar si la funcion no es correcta, en cuyo caso se lanza una excepcion
    if not in_func in functions:
        raise ValueError('Error: la funcion tiene que ser una de las siguientes: {}'.format(functions))

    # Determinar archivo CSV de entrada
    csv_file = 'data/' + in_file + '.csv'

    # Cargar el archivo csv que contiene los datos y obtener la muestra
    df = pd.read_csv(csv_file)
    sample = df.values[:, 1:]

    # Obtener los valores x, y de la muestra (normalizar x)
    sample_x = normalize_data(sample[:, :-1])
    sample_y = sample[:, -1].reshape(-1,)

    # Dividir la muestra en particiones disjuntas 
    # Se obtienen 5 particiones, organizadas en 2 listas 
    # La primera contiene 5 particiones de entrenamiento con sus (x, y)
    # La segunda contiene 5 particiones de test con sus (x, y)
    train_part, test_part = stratify_sample(sample_x, sample_y)

    # Decidir que funcion se llama
    if in_func == 'knn':
        print('Clasificador: KNN')
        classifier = knn_classifier
    elif in_func == 'relief':
        print('Clasificador: RELIEF')
        classifier = relief_classifier
    else:
        print('Clasificador: Busqueda Local')
        classifier = local_search_classifier

    print('Conjunto de datos: {}'.format(in_file))
    # Llamar a la funcion de clasificacion y combinar los datos
    class_rate, red_rate, aggrupation, times = classifier(train_part, test_part)

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
    print('\nTiempo total: {}'.format(times.sum()))
