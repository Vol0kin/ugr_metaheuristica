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
#                        Operador de mutacion                                 #
###############################################################################

def mutation_operator(chromosome, gen):
    """
    Funcion para generar una mutacion en el gen de un cromosoma

    :param chromosome: Cromosoma a mutar
    :param gen: Gen del cromosoma a mutar
    """

    # Aplicar mutacion (sumarle valor generado por una normal de media 0 
    # y desviacion tipica 0.3)
    chromosome[gen] += np.random.normal(0.0, 0.3)

    # Normalizar el cromosoma
    np.clip(chromosome, 0.0, 1.0)


###############################################################################
#                   Torneo binario de seleccion                               #
###############################################################################

def binary_tournament(pop_fitness, parents):
    """
    Funcion que simula un torneo binario entre dos padres para determinar el
    mejor de ellos

    :param pop_fitness: Valors fitness de la poblacion
    :param parents: Indices de los padres que comparar

    :return Devuelve el mejor padre segun la funcion fitness
    """

    # Obtener los valores fitness de los padres
    parents_fitness = pop_fitness[parents]

    # Elegir como mejor padre el primero
    best_parent = parents[0]

    # Comprobar si el segundo padre es mejor que el primero
    if parents_fitness[0] < parents_fitness[1]:
        best_parent = parents[1]

    return best_parent


###############################################################################
#                           Ordenar la poblacion                              #
###############################################################################

def sort_population(fitness_values, population):
    """
    Funcion para ordenar la poblacion segun su valor de la funcion fitness

    :param fitness_values: Lista de valores fitness
    :param population: Conjunto de cromosomas que ordenar
    """

    # Obtener los indices ordenados de los valores fitness
    # en orden descendente (de mayor a menor)
    index = np.argsort(fitness_values)[::-1]

    # Ordenar valores fitness y poblacion
    fitness_values = fitness_values[index]
    population = population[index]


###############################################################################
#               Implementacion de los algoritmos geneticos                    #
###############################################################################

def genetic_algorithm(data, labels, cross_rate, mutation_rate, cross_func,
                      chromosomes=30, max_evals=15000):
    
    
    genes = data.shape[1]
    
    if cross_func == blx_alpha_crossover:
        expected_crosses = int(chromosomes / 2 * cross_rate)
    else:
        expected_crosses = int(chromosomes * cross_rate)

    expected_mutations = chromosomes * genes * mutation_rate

    n_evaluations = 0

    # Generar poblacion inicial y evaluarla
    population = generate_initial_population(chromosomes, genes)
    pop_fitness = metrics.evaluate_population(data, labels, population)

    n_evaluations += chromosomes

    # Ordenar poblacion por valor fitness
    sort_population(pop_fitness, population)

    while n_evaluations < max_evals:

        # Crear una lista de padres
        parents_list = []

        # Realizar numero de cromosomas torneos binarios para determinar
        # quienes seran los padres
        for _ in range(chromosomes):

            # Elegir dos cromosomas aleatorios
            parents = np.random.choice(chromosomes, 2)

            # Realizar el torneo binario
            parents_list.append(binary_tournament(pop_fitness, parents))

        # Convertir padres obtenidos a array
        parents = np.array(parents_list)

        # Obtener los padres que participaran en el cruce
        cross_parents = parents.reshape(-1, 2)[:expected_crosses, :]

        # Aplicar operador de cruce para obtener los descendientes
        offspring = cross_func(parents, population)



    return population[0]
