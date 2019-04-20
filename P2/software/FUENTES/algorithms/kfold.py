import numpy as np
from sklearn.model_selection import StratifiedKFold         # Particionar muestra

# Modulo para la normalizacion y creacion de las muestras

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
