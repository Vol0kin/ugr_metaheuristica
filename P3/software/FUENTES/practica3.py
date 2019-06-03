import numpy as np
import pandas as pd
import sys                                                  # Argumentos de la linea de comandos
import algorithms.kfold
import algorithms.classifiers

# Posibles archivos y algoritmos
files = ('colposcopy', 'ionosphere', 'texture')
algorithm_list = ('sa', 'ils', 'der', 'de')

# Leer archivo y algoritmo de entrada
in_file = sys.argv[1]
in_algorithm = sys.argv[2]

# Comprobar si el archivo no es correcto, en cuyo caso se lanza una excepcion
if not in_file in files:
    raise ValueError('Error: el archivo tiene que ser uno de los siguientes: {}'.format(files))

# Comprobar si la funcion no es correcta, en cuyo caso se lanza una excepcion
if not in_algorithm in algorithm_list:
    raise ValueError('Error: la funcion tiene que ser una de las siguientes: {}'.format(algorithm_list))

# Determinar archivo CSV de entrada
csv_file = '../BIN/' + in_file + '.csv'

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

# Crear la lista de argumentos del clasificador (se descomprimiran luego)
classifier_args = []

# Decidir el algoritmo a ejecutar en la busqueda
if 'sa' in in_algorithm:
    algo = algorithms.classifiers.SearchAlgorithm.SA
elif 'ils' in in_algorithm:
    algo = algorithms.classifiers.SearchAlgorithm.ILS
elif 'der' in in_algorithm:
    algo = algorithms.classifiers.SearchAlgorithm.DER
else:
    algo = algorithms.classifiers.SearchAlgorithm.DE

print('Conjunto de datos: {}'.format(in_file))
print('Clasificador utilizado: {}'.format(algo))

# Establecer la semilla aleatoria
np.random.seed(8912374)

class_rate, red_rate, aggrupation, times = algorithms.classifiers.classifier(train_part, test_part, algo)


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
