import numpy as np
from . import metrics
from . import utils

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


def differential_evolution(X, y, use_best, pop_size=50, cr=0.5, f=0.5,
                           max_evaluations=15000):
    """
    Funcion que simula la Evolucion Diferencial para el calculo de un
    vector de pesos

    :param X: Conjunto de datos
    :param y: Etiquetas
    :param pop_size: Numero de habitantes de la poblacion (por defecto 50)
    :param cr: Tasa de cruce (por defecto 0.5)
    :param f: Ponderacion que se le da a una combinacion en un cruce (por
              defecto 0.5)
    :param max_evaluations: Numero maximo de evaluaciones de la funcion
                            objetivo (por defecto 15000)

    :return Devuelve el mejor cromosoma de la poblacion
    """

    # Obtener numero de caracteristicas
    N = X.shape[1]

    # Establecer numero inicial de evaluaciones
    n_evals = 0

    # Establecer numero de padres que se utilizaran
    if not use_best:
        n_parents = 3
    else:
        n_parents = 2
    
    # Generar poblacion inicial y evaluarla
    population = generate_initial_population(pop_size, N)
    pop_fitness = metrics.evaluate_population(X, y, population)

    n_evals += pop_size

    # Ordenar la poblacion por valor fitness
    pop_fitness, population = sort_population(pop_fitness, population)

    while n_evals < max_evaluations:
        
        # Crear lista de descendientes
        offsprings = []
        
        # Recorrer la poblacion y aplicar evolucion diferencial
        for i in range(pop_size):
            # Generar indices de los posibles padres
            parent_idx = [j for j in range(pop_size) if j != i]

            # Escoger un numero de padres aleatorios
            parents = np.random.choice(parent_idx, n_parents, replace=False)

            # Generar probabilidades para cada gen de cruzar
            prob_array = np.random.uniform(0.0, 1.0, N)

            # Obtener indices donde cruzar y donde no
            cross_idx = np.where(prob_array < cr)
            individual_idx = np.where(prob_array >= cr)
#            print(cross_idx)
#            print(individual_idx)

            # Generar descendiente
            offspring = np.empty_like(population[i])
            
            if not use_best:
                offspring[cross_idx] = population[parents[0], cross_idx] + f * (population[parents[1], cross_idx] - population[parents[2], cross_idx])
            else:
                offspring[cross_idx] = population[0, cross_idx] + f * (population[0, cross_idx] - population[i, cross_idx]) + f * (population[parents[0], cross_idx] - population[parents[1], cross_idx])

            offspring[individual_idx] = population[i, individual_idx]

            # Normalizar descendiente
            offspring = utils.normalize_w(offspring)

            # Insertar descendiente
            offsprings.append(offspring)

        # Obtener evaluacion de los hijos
        offsprings = np.array(offsprings)
        offspring_fit = metrics.evaluate_population(X, y, offsprings)

        n_evals += pop_size

        # Intercambio one-to-one
        best_offsprings = np.where(offspring_fit > pop_fitness)
        population[best_offsprings] = offsprings[best_offsprings]
        pop_fitness[best_offsprings] = offspring_fit[best_offsprings]

        # Ordenar nueva poblacion
        pop_fitness, population = sort_population(pop_fitness, population)

    return population[0]
