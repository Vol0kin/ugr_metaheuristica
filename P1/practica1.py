import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pykdtree.kdtree import KDTree

def normalize_data(sample):
    # Obtener minimo y maximo de cada columna de la muestra
    # y la diferencia entre estos
    sample_min = sample.min(axis=0)
    sample_max = sample.max(axis=0)
    sample_diff = sample_max - sample_min

    # Normalizar los datos 
    new_sample = (sample - sample_min) / sample_diff

    return new_sample

def stratify_sample(X, y):
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


df = pd.read_csv('data/colposcopy.csv')
sample = df.values[:, 1:]

sample_x = sample[:, :-1]
sample_y = sample[:, -1].reshape(-1,)

print(np.where(sample_y == 0.)[0].shape)
print(np.where(sample_y == 1.)[0].shape)

sample_x = normalize_data(sample_x)

train_part, test_part = stratify_sample(sample_x, sample_y)


