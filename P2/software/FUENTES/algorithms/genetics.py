import numpy as np
from . import metrics
from . import utils

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

    # Normalizar valores
    children = utils.normalize_w(children)

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

def mutation_operator(population, chromosomes, genes, mean=0.0, sigma=0.3):
    """
    Funcion para generar una mutacion en el gen de un cromosoma

    :param population: Poblacion que mutar
    :param chromosomes: Cromosomas a mutar
    :param genes: Genes de los cromosomas a mutar
    :param mean: Media de la mutacion (por defecto 0.0)
    :param sigma: Desviacion tipica de la mutacion (por defecto 0.3)
    
    :return Devuelve la poblacion mutada
    """
    
    # Aplicar mutacion (sumarle valor generado por una normal de media 0
    # y desviacion tipica 0.3)
    population[chromosomes, genes] += np.random.normal(mean, sigma, chromosomes.shape[0])

    # Normalizar el/los cromosoma/s mutados
    population = utils.normalize_w(population)
    
    return population


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
    
    :return Devuelve los valores fitnes y la poblacion ordenados por fitness
    """

    # Obtener los indices ordenados de los valores fitness
    # en orden descendente (de mayor a menor)
    index = np.argsort(fitness_values)[::-1]

    # Ordenar valores fitness y poblacion
    sorted_fitness = fitness_values[index]
    sorted_population = population[index]

    return sorted_fitness, sorted_population


###############################################################################
#                               Elitismo                                      #
###############################################################################

def elitism(old_fitness, old_population, new_fitness, new_population):
    """
    Funcion para aplicar el criterio de elitismo en un algoritmo genetico
    generacional. El elitismo solo se aplica si el mejor cromosoma de la
    poblacion anterior es mejor que el mejor cromosoma de la nueva poblacion,
    con el objetivo de no perderlo

    :param old_fitness: Antiguos valores fitness
    :param old_population: Antigua poblacion
    :param new_fitness: Nuevos valores fitness
    :param new_population: Nueva poblacion

    :return Devuleve los nuevos valores fitness y poblacion al aplicar el
            criterio de elitismo
    """

    # Obtener mejores valores fitness de las dos poblaciones
    old_best_fit = old_fitness[0]
    new_best_fit = new_fitness[0]

    # Si se va a perder el mejor cromosoma, se le reintroduce en la nueva
    # poblacion eliminando el ultimo de la nueva
    if old_best_fit > new_best_fit:
        # Insertar en la poblacion y eliminar el ultimo
        new_population = np.r_[old_population[0].reshape(1, -1), new_population]
        new_population = np.delete(new_population, -1, axis=0)

        # Insertar valor fitness y eliminar el ultimo
        new_fitness = np.r_[old_best_fit, new_fitness]
        new_fitness = np.delete(new_fitness, -1)

    return new_fitness, new_population


###############################################################################
#               Implementacion de los algoritmos geneticos                    #
###############################################################################

def generational_genetic_algorithm(data, labels, cross_func, cross_rate=0.7,
                                   mutation_rate=0.001, chromosomes=30,
                                   max_evals=15000):
    """
    Implementacion del algoritmo genetico generacional con los operadores de
    cruce BLX-alpha y cruce aritmetico. Se parte de una poblacion inicial
    generada aleatoriamente y se va refinando hasta obtener unos buenos pesos
    para el problema de clasificacion.

    :param data: Conjunto de datos
    :param labels: Conjunto de etiquetas
    :param cross_func: Funcion de cruce. Puede ser BLX-alpha o cruce aritmetico
    :param cross_rate: Probabilidad de cruce (por defecto 0.7)
    :param mutation_rate: Probabilidad de mutar (por defecto 0.001)
    :param chromosomes: Numero de cromosomas iniciales (por defecto 30)
    :param max_evals: Numero maximo de evaluaciones de la funcion objetivo
                      (por defecto 15000)

    :return Devuelve el mejor w de la poblacion despues de 15000 evaluaciones
            de la funcion objetivo
    """

    # Obtener numero de genes
    genes = data.shape[1]

    # Obtener numero esperado de cruces, mutaciones y numero de padres
    # Los valores cambiaran segun la funcion de cruce utilizada
    if cross_func == blx_alpha_crossover:
        expected_crosses = int(chromosomes / 2 * cross_rate)
        n_parents = chromosomes
    else:
        # Se deben realizar mas cruces debido a que el operador solo produce
        # un descendiente (el doble que con BLX-alpha)
        expected_crosses = int(chromosomes * cross_rate)
        n_parents = expected_crosses + chromosomes

    expected_mutations = chromosomes * genes * mutation_rate

    # Establecer cuando mutar una generacion (se ira acumulando hasta que
    # tenga un valor mayor o igual a uno, y eso indicara que se debe
    # mutar en esa generacion)
    mutate_generation = expected_mutations

    # Establecer frontera entre padres que se cruzan y padres que no
    # La frontera viene dada por el doble del numero de parejas
    index_limit_cross_copy = expected_crosses * 2

    # Inicializar las evaluaciones
    n_evaluations = 0

    # Generar poblacion inicial y evaluarla
    population = generate_initial_population(chromosomes, genes)
    pop_fitness = metrics.evaluate_population(data, labels, population)

    n_evaluations += chromosomes

    # Ordenar poblacion por valor fitness
    pop_fitness, population = sort_population(pop_fitness, population)

    while n_evaluations < max_evals:
        # Crear array de modificados
        # 0 -> el cromosoma no se ha obtenido por cruce, es igual al de la
        # poblacion anterior
        # 1 -> el cromosoma se ha obtenido por cruce o por mutacion
        modified = np.zeros((chromosomes,), dtype=np.int)
        
        # Crear una lista de padres
        parents_list = []

        # Realizar tantos torneos binarios como cromosomas se tengan
        # que generar 
        for _ in range(n_parents):
            # Elegir dos cromosomas aleatorios
            parents = np.random.choice(chromosomes, 2)

            # Realizar el torneo binario
            parents_list.append(binary_tournament(pop_fitness, parents))

        parents = np.array(parents_list)

        # Formar las parejas de padres que se van a cruzar
        cross_parents = parents[:index_limit_cross_copy].reshape(-1, 2)

        # Aplicar operador de cruce para obtener los descendientes
        offspring = cross_func(cross_parents, population)

        # Marcar los cromosomas que se han obtenido por cruce
        if cross_func == blx_alpha_crossover:
            modified[: index_limit_cross_copy] = 1
        else:
            modified[: expected_crosses] = 1

        # Obtener los descendientes sin cruzar
        offspring_no_cross = population[parents[index_limit_cross_copy:], :]

        # Generar nueva poblacion
        new_population = np.r_[offspring, offspring_no_cross]

        # Proceso de mutacion
        if mutate_generation >= 1.0:
            # Obtener numero de mutaciones (truncando)
            n_mutations = int(mutate_generation)

            # Reiniciar contador de mutaciones
            mutate_generation = expected_mutations

            # Generar cromosmas y genes a mutar
            mut_chromosome = np.random.choice(chromosomes, n_mutations, replace=True)
            mut_gene = np.random.choice(genes, n_mutations)

            # Aplicar mutacion
            new_population = mutation_operator(new_population, mut_chromosome, mut_gene)

            # Indicar que se han modificado los cromosomas correspondientes
            modified[mut_chromosome] = 1
        else:
            # Incrementar contador de mutaciones
            mutate_generation += expected_mutations

        # Evaluar la poblacion
        new_pop_fitness = []

        # Recorrer la lista de modificados y ver si se tiene que evaluar de
        # nuevo el cromosoma o no
        for i in range(chromosomes):
            if modified[i] == 1:
                # Evaluar nuevo cromosoma
                new_pop_fitness.append(metrics.evaluate(data, labels, new_population[i]))
            else:
                # Como no se ha modificado el cromosoma, se obtiene su valor
                # fitness de la poblacion anterior
                fitness_index = i

                # Si se utiliza cruce aritmetico, se debe aplicar un desplazamiento
                # para poder obetener el indice (debido a que hay mas padres, se tiene
                # que aplicar un desplazamiento equivalente al numero esperado de cruces)
                if cross_func == arithmetic_crossover:
                    fitness_index += expected_crosses

                # Obetener valor fitness de la poblacion anterior
                new_pop_fitness.append(pop_fitness[parents[fitness_index]])

        new_pop_fitness = np.array(new_pop_fitness)

        # Actualizar el numero de evaluaciones realizadas
        n_evaluations += modified.sum()

        # Ordenar la nueva poblacion por fitness
        new_pop_fitness, new_population = sort_population(new_pop_fitness, new_population)
        
        # Elitismo y sustitucion generacional de la poblacion
        pop_fitness, population = elitism(pop_fitness, population, new_pop_fitness, new_population)

    return population[0]


def stationary_genetic_algorithm(data, labels, cross_func, cross_rate=1.0,
                                 mutation_rate=0.001, chromosomes=30,
                                 max_evals=15000):
    """
    Implementacion del algoritmo genetico estacionario con los operadores de
    cruce BLX-alpha y cruce aritmetico. Se parte de una poblacion inicial
    generada aleatoriamente y se va refinando hasta obtener unos buenos pesos
    para el problema de clasificacion.

    :param data: Conjunto de datos
    :param labels: Conjunto de etiquetas
    :param cross_func: Funcion de cruce. Puede ser BLX-alpha o cruce aritmetico
    :param cross_rate: Probabilidad de cruce (por defecto 0.7)
    :param mutation_rate: Probabilidad de mutar (por defecto 0.001)
    :param chromosomes: Numero de cromosomas iniciales (por defecto 30)
    :param max_evals: Numero maximo de evaluaciones de la funcion objetivo
                      (por defecto 15000)

    :return Devuelve el mejor w de la poblacion despues de 15000 evaluaciones
            de la funcion objetivo
    """

    # Obtener numero de genes
    genes = data.shape[1]

    # Establecer numero de hijos (cromosomas)
    n_children = 2

    # Establecer numero de padres y de mutaciones esperadas por generacion
    # Los valores cambiaran segun la funcion de cruce utilizada
    if cross_func == blx_alpha_crossover:
        n_parents = n_children
    else:
        # Se deben realizar mas cruces debido a que el operador solo produce
        # un descendiente (el doble que con BLX-alpha)
        n_parents = n_children * 2

    expected_mutations = n_children * genes * mutation_rate

    # Establecer cuando mutar una generacion (se ira acumulando hasta que
    # tenga un valor mayor o igual a uno, y eso indicara que se debe
    # mutar en esa generacion)
    mutate_generation = expected_mutations

    # Inicializar las evaluaciones
    n_evaluations = 0

    # Generar poblacion inicial y evaluarla
    population = generate_initial_population(chromosomes, genes)
    pop_fitness = metrics.evaluate_population(data, labels, population)

    n_evaluations += chromosomes

    # Ordenar poblacion por valor fitness
    pop_fitness, population = sort_population(pop_fitness, population)

    while n_evaluations < max_evals:
        # Crear una lista de padres
        parents_list = []

        # Realizar tantos torneos binarios como cromosomas se tengan
        # que generar 
        for _ in range(n_parents):

            # Elegir dos cromosomas aleatorios
            parents = np.random.choice(chromosomes, 2)

            # Realizar el torneo binario
            parents_list.append(binary_tournament(pop_fitness, parents))

        parents = np.array(parents_list)

        # Formar las parejas de padres que se van a cruzar
        cross_parents = parents.reshape(-1, 2)

        # Aplicar operador de cruce para obtener los descendientes
        offspring = cross_func(cross_parents, population)

        # Proceso de mutacion
        if mutate_generation >= 1.0:
            # Obtener numero de mutaciones (truncando)
            n_mutations = int(mutate_generation)

            # Reiniciar contador de mutaciones
            mutate_generation = expected_mutations

            # Generar cromosmas y genes a mutar
            mut_chromosome = np.random.choice(n_children, n_mutations, replace=True)
            mut_gene = np.random.choice(genes, n_mutations)

            # Aplicar mutacion
            new_population = mutation_operator(offspring, mut_chromosome, mut_gene)
        else:
            # Incrementar contador de mutaciones
            mutate_generation += expected_mutations

        # Evaluar los descendientes y ordenarlos por fitness
        offspring_fitness = metrics.evaluate_population(data, labels, offspring)
        offspring_fitness, offspring = sort_population(offspring_fitness, offspring)

        # Incrementar el numero de evaluaciones
        n_evaluations += n_children

        # Crear poblacion y fitness de torneo, compuesto por los dos ultimos
        # elementos de la poblacion y por los dos descendientes, en este orden
        pop_tournament = np.r_[population[-2:, :], offspring]
        tournament_fitness = np.r_[pop_fitness[-2:], offspring_fitness]

        # Ordenar ultimos dos elementos de la poblacion y descendientes por
        # el valor de fitness obtenido para estos
        tournament_fitness, pop_tournament = sort_population(tournament_fitness, pop_tournament)

        # Insertar en los dos ultimos lugares aquellos que tienen mejor fitness
        population[-2:] = pop_tournament[:2]
        pop_fitness[-2:] = tournament_fitness[:2]

        # Ordenar de nuevo la poblacion, en caso de que hayan habido cambios
        # significativos
        pop_fitness, population = sort_population(pop_fitness, population)

    return population[0]


def genetic_algorithm(data, labels, cross_func, generational=True):
    """
    Funcion que sirve como interfaz para llamar a los dos algoritmos geneticos

    :param data: Conjunto de datos
    :param labels: Conjunto de etiquetas
    :param cross_func: Funcion de cruce a utilizar
    :param generational: Utilizar o no un enfoque generacional (por defecto True)

    :return Devuelve el mejor w de la poblacion segun alguno de los dos criterios
    """
    if generational == True:
        return generational_genetic_algorithm(data, labels, cross_func)
    else:
        return stationary_genetic_algorithm(data, labels, cross_func)
