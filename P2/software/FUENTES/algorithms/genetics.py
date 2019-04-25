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

    children[children < 0.0] = 0.0
    children[children > 1.0] = 1.0

#    np.clip(children, 0.0, 1.0)

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

def mutation_operator(population, chromosomes, genes):
    """
    Funcion para generar una mutacion en el gen de un cromosoma

    :parm population: Poblacion que mutar
    :param chromosomes: Cromosomas a mutar
    :param genes: Genes de los cromosomas a mutar
    """

    # Aplicar mutacion (sumarle valor generado por una normal de media 0
    # y desviacion tipica 0.3)

    population[chromosomes, genes] += np.random.normal(0.0, 0.3, chromosomes.shape[0])

    # Normalizar el cromosoma
    population[population < 0.0] = 0.0
    population[population > 1.0] = 1.0
    #np.clip(population, 0.0, 1.0)


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
    sorted_fitness = fitness_values[index]
    sorted_population = population[index]

    return sorted_fitness, sorted_population


###############################################################################
#               Implementacion de los algoritmos geneticos                    #
###############################################################################

def genetic_algorithm(data, labels, cross_rate, mutation_rate, cross_func,
                      chromosomes=30, max_evals=15000):

    # Obtener numero de genes
    genes = data.shape[1]
    print("numero de genes ", genes)
    
    # Obtener numero esperado de cruces y mutaciones
    if cross_func == blx_alpha_crossover:
        expected_crosses = int(chromosomes / 2 * cross_rate)
    else:
        expected_crosses = int(chromosomes * cross_rate)

    expected_mutations = chromosomes * genes * mutation_rate

    # Establecer cuando mutar una generacion
    mutate_generation = expected_mutations

    # Establecer indice de los padres que no cruzan
    index_no_cross = expected_crosses * 2

    print("numero esperado de cruces: ", expected_crosses)
    print("numero esperado de mutaciones: ", expected_mutations)

    # Inicializar las evaluaciones
    n_evaluations = 0

    # Generar poblacion inicial y evaluarla
    population = generate_initial_population(chromosomes, genes)
    pop_fitness = metrics.evaluate_population(data, labels, population)

    n_evaluations += chromosomes

    # Ordenar poblacion por valor fitness
    pop_fitness, population = sort_population(pop_fitness, population)

    while n_evaluations < max_evals:
        print("evaluaciones al comienzo del bucle ", n_evaluations)
        # Crear una lista de padres
        parents_list = []

        # Realizar tantos torneos binarios como cromosomas haya para
        # decidir quienes seran los padres
        for _ in range(chromosomes):

            # Elegir dos cromosomas aleatorios
            parents = np.random.choice(chromosomes, 2)

            #print("padres: ", parents)

            # Realizar el torneo binario
            parents_list.append(binary_tournament(pop_fitness, parents))

        # Convertir padres obtenidos a array
        parents = np.array(parents_list)

        # Obtener los padres que participaran en el cruce
        cross_parents = parents.reshape(-1, 2)[:expected_crosses, :]

        # Aplicar operador de cruce para obtener los descendientes
        offspring = cross_func(cross_parents, population)
        print("numero de descendientes: ", offspring.shape)

        # Obtener los descendientes sin cruzar
        offspring_no_cross = population[parents[index_no_cross:], :]
        print("numero de descendientes sin cruce: ", offspring_no_cross.shape)

        # Combinar los dos en la nueva poblacion 
        new_population = np.r_[offspring, offspring_no_cross]

        # Proceso de mutacion
        if mutate_generation >= 1.0:
            n_mutations = int(mutate_generation)
            mutate_generation = expected_mutations

            mut_chromosome = np.random.choice(chromosomes, n_mutations, replace=True)
            mut_gene = np.random.choice(genes, n_mutations)

            print(mut_chromosome)
            print(mut_gene)

            mutation_operator(new_population, mut_chromosome, mut_gene)
        else:
            mutate_generation += expected_mutations

        # Evaluar la nueva poblacion
        new_pop_fitness = metrics.evaluate_population(data, labels, new_population)
        n_evaluations += chromosomes

        # Ordenar la nueva poblacion por fitness
        new_pop_fitness, new_population = sort_population(new_pop_fitness, new_population)

        # Proceso de elitismo
        if pop_fitness[0] > new_pop_fitness[-1]:
            print("elitismo")
            new_pop_fitness[-1] = pop_fitness[0]
            new_population[-1] = population[0]
            new_pop_fitness, new_population = sort_population(new_pop_fitness, new_population)

        # Aplicar criterio generacional
        population = new_population
        pop_fitness = new_pop_fitness

        print(population)
        print(pop_fitness)

    return population[0]
