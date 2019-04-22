import numpy as np
from . import metrics

###############################################################################
#                     Generacion de poblacion inicial                         #
###############################################################################

def generate_initial_population(chromosomes, genes):
    """
    Funcion para generar una nueva poblacion inicial

    :param chromosomes: Numero de comosomas a generar
    :param genes: Numero de genes por cromosoma

    :return Devuelve una nueva poblacion formada por valores aleatorios entre
            [0, 1] de dimension cromosomas x genes
    """

    population = np.random.uniform(0.0, 1.0, size=(chromosomes, genes))

    return population


###############################################################################
#                           Operadores de cruce                               #
###############################################################################

def blx_alpha_crossover(parents, chromosomes, alpha=0.3):
    """
    Funcion para la generacion de descendientes mediante el operador de cruce
    BLX-alpha

    :param parents: Matriz de parejas de padres
    :param chromosomes: Conjunto de cromosomas
    :param alpha: Valor alpha del operador. Por defecto 0.3.

    :return Devuelve una matriz de hijos generados mediante el operador de
            cruce
    """

    # Obtener los cromosomas de los padres
    parents_chromosomes = chromosomes[parents, :]

    # Obtener los valores c_min y c_max para cada pareja
    # de caracteristicas
    c_min = parents_chromosomes.min(axis=1)
    c_max = parents_chromosomes.max(axis=1)

    # Replicar cada valor minimo y maximo para tenerlo
    # dos veces seguidas
    c_min = np.repeat(c_min, 2, axis=0)
    c_max = np.repeat(c_max, 2, axis=0)

    # Calcular amplitud del intervalo
    i = c_max - c_min

    # Generar hijos y normalizarlos al rango [0, 1]
    children = np.random.uniform(c_min - i * alpha, c_max + i * alpha)
    np.clip(children, 0.0, 1.0)

    return children


def arithmetic_crossover(parents, chromosomes):
    """
    Funcion para la generacion de descendientes mediante el operador de cruce 
    aritmetico

    :param parents: Matriz 2D de padres
    :param chromosomes: Lista de cromosomas

    :return Devuelve una matriz de hijos generados mediante el operador de 
            cruce
    """

    # Seleccionar todos los cromosomas de los padres
    parent_chromosomes = chromosomes[parents, :]

    # Obtener los hijos haciendo la media aritmetica
    # Aplicar sobre eje 1 (son multiples matrices)
    children = parent_chromosomes.mean(axis=1)

    return children


###############################################################################
#                   Torneo binario de seleccion                               #
###############################################################################

def binary_tournament(data, labels, parents, chromosomes):
    """
    Funcion que simula un torneo binario entre dos padres para determinar el
    mejor de ellos

    :param data: Conjunto de vectores de caracteristicas
    :param labels: Conjunto de etiquetas, una por cada fila de datos
    :param parents: Indices de los padres que comparar
    :param chromosomes: Conjunto de cromosomas

    :return Devuelve el mejor padre segun la funcion fitness
    """
    # Obtener los cromosomas de los padres
    parents_chromosomes = chromosomes[parents, :]

    # Calcular valors fitness de los padres
    fit_value_parent1 = metrics.evaluate(data, labels, parents_chromosomes[0])
    fit_value_parent2 = metrics.evaluate(data, labels, parents_chromosomes[1])

    # Elegir como mejor padre el primero
    best_parent = parents[0]

    # Comprobar si el segundo padre es mejor que el primero
    if fit_value_parent1 < fit_value_parent2:
        best_parent = parents[1]

    return best_parent


###############################################################################
#               Implementacion de los algoritmos geneticos                    #
###############################################################################

def genetic_algorithm(data, labels, cross_rate, mutation_rate, cross_func,
                      chromosomes=30, max_evals=15000):
    
    
    n_gens = data.shape[1]

    expected_crosses = chromosomes / 2 * cross_rate
    expected_mutations = chromosomes * n_gens * mutation_rate

    n_evaluations = 0

    while n_evaluations < max_evals:
        None

    return None
