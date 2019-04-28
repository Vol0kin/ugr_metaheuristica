import numpy as np
from . import genetics
from . import utils
from . import metrics

###############################################################################
#                     Mutacion en la busqueda local                           #
###############################################################################

def neighbor_mutation(w, gene, mean=0.0, sigma=0.3):
    """
    Funcion para generar una mutacion para explorar el vecindario del elemento

    :param w: Vector de pesos sobre el que aplicar la mutacion
    :param gene: Elemento del vector sobre el que mutar
    :param mean: Media de la mutacion (por defecto 0.0)
    :param sigma: Desviacion tipica de la mutacion (por defecto 0.3)
    """

    # Aplicar mutacion (sumarle valor generado por una normal de media 0
    # y desviacion tipica 0.3)
    w[gene] += np.random.normal(mean, sigma)

    # Normalizar el/los cromosoma/s mutados
    w = utils.normalize_w(w)

    return w


###############################################################################
#                            Busqueda local                                   #
###############################################################################

def local_search(data, labels, initial_w, initial_fit, max_evaluations):
    """
    Funcion para el calculo de w mediante la busqueda local

    :param data: Conjunto de datos
    :param labels: Conjunto de caracteristicas
    :param initial_w: Valor inicial de w
    :param initial_fit: Valor fitness inicial
    :param max_evaluations: Maximo numero de evaluaciones de la funcion fitness
                            que puede hacer la busqueda

    :return Devuelve el w obtenido despues del proceso de busqueda local junto
            con su valor fitness
    """

    # Obtener numero de caracteristicas
    N = data.shape[1]

    # Copiar w inicial y valor fitness inicial
    w = np.copy(initial_w)
    fitness_val = initial_fit

    # Inicializar evaluaciones
    evaluations = 0

    # Mientras no se hayan superado el numero maximo de evaluaciones
    # intentar ajustar w
    while evaluations < max_evaluations:
        # Copiar w
        current_w = np.copy(w)

        # Intentar mutar cada caracteristica en el orden dado por la permutacion
        # hasta encontrar la primera mutacion que obtiene un mejor valor de fitness
        for trait in np.random.permutation(N):
            # Mutar el w_i con un valor de la normal con media mean y d.t. sigma
            w = neighbor_mutation(w, trait)

            # Evaluar el nuevo w con los datos
            evaluations += 1
            new_fitness = metrics.evaluate(data, labels, w)

            # Si el nuevo valor de fitness es mejor, conservar w, guardar el mejor
            # valor de fitness e indicar que se ha realizado una evaluacion con
            # exito
            if new_fitness > fitness_val:
                fitness_val = new_fitness
                break
            # En caso contrario, contabilizar un error mas y restaurar los w
            else:
                w[trait] = current_w[trait]

            # Si se sobrepasa el numero max de evaluaciones o se obtienen demasiados
            # errores, se termina la busqueda
            if evaluations > max_evaluations:
                return w, fitness_val


    return w, fitness_val


###############################################################################
#                         Algoritmos memeticos                                #
###############################################################################

def memetic_algorithm(data, labels, ls_rate, cross_rate=0.7, mutation_rate=0.001,
                      chromosomes=30, max_evals=15000, ls_best=False):
    """
    Implementacion del algoritmo memetico basandose en el algoritmo genetico
    generacional con el operador de cruce BLX-alpha. Se parte de una poblacion
    inicial generada aleatoriamente y se va refinando hasta obtener unos buenos
    pesos para el problema de clasificacion.

    :param data: Conjunto de datos
    :param labels: Conjunto de etiquetas
    :param ls_rate: Tasa de elementos de la poblacion sobre los que aplicar la
    								busqueda local
    :param cross_rate: Probabilidad de cruce (por defecto 0.7)
    :param mutation_rate: Probabilidad de mutar (por defecto 0.001)
    :param chromosomes: Numero de cromosomas iniciales (por defecto 30)
    :param max_evals: Numero maximo de evaluaciones de la funcion objetivo
                      (por defecto 15000)
    :param ls_best: Indica si aplicar la busqueda local sobre los ls_rate mejores
    								cromosomas de la poblacion (por defecto False)

    :return Devuelve el mejor w de la poblacion despues de 15000 evaluaciones
            de la funcion objetivo
    """

    # Obtener numero de genes
    genes = data.shape[1]

    # Establecer numero esperado de cruces, mutaciones y evals. en local search
    expected_crosses = int(chromosomes / 2 * cross_rate)
    expected_mutations = chromosomes * genes * mutation_rate
    ls_evals = genes * 2

    # Establecer cuando mutar una generacion (se ira acumulando hasta que
    # tenga un valor mayor o igual a uno, y eso indicara que se debe
    # mutar en esa generacion)
    mutate_generation = expected_mutations

    # Establecer frontera entre padres que se cruzan y padres que no
    # La frontera viene dada por el doble del numero de parejas
    index_limit_cross_copy = expected_crosses * 2

    # Inicializar las evaluaciones y generaciones
    n_evaluations = 0
    n_generations = 0

    # Generar poblacion inicial y evaluarla
    population = genetics.generate_initial_population(chromosomes, genes)
    pop_fitness = metrics.evaluate_population(data, labels, population)

    n_evaluations += chromosomes
    n_generations += 1

    # Ordenar poblacion por valor fitness
    pop_fitness, population = genetics.sort_population(pop_fitness, population)

    while n_evaluations < max_evals:
        # Crear array de modificados
        # 0 -> el cromosoma no se ha obtenido por cruce, es igual al de la
        # poblacion anterior
        # 1 -> el cromosoma se ha obtenido por cruce o por mutacion
        modified = np.zeros((chromosomes,), dtype=np.int)

        print("Evaluaciones al comienzo del bucle ", n_evaluations)
        # Crear una lista de padres
        parents_list = []

        # Realizar tantos torneos binarios como cromosomas se tengan
        # que generar 
        for _ in range(chromosomes):

            # Elegir dos cromosomas aleatorios
            parents = np.random.choice(chromosomes, 2)

            # Realizar el torneo binario
            parents_list.append(genetics.binary_tournament(pop_fitness, parents))

        parents = np.array(parents_list)

        # Formar las parejas de padres que se van a cruzar
        cross_parents = parents[:index_limit_cross_copy].reshape(-1, 2)

        # Aplicar operador de cruce para obtener los descendientes
        offspring = genetics.blx_alpha_crossover(cross_parents, population)

        # Marcar los cromosomas que se han obtenido por cruce
        modified[: index_limit_cross_copy] = 1
        print("Numero de descendientes: ", offspring.shape)

        # Obtener los descendientes sin cruzar
        offspring_no_cross = population[parents[index_limit_cross_copy:], :]
        print("Numero de descendientes sin cruce: ", offspring_no_cross.shape)

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

            print(mut_chromosome)
            print(mut_gene)

            # Aplicar mutacion
            new_population = genetics.mutation_operator(new_population, mut_chromosome, mut_gene)

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
                new_pop_fitness.append(pop_fitness[parents[i]])

        new_pop_fitness = np.array(new_pop_fitness)

        # Actualizar el numero de evaluaciones realizadas
        n_evaluations += modified.sum()
        n_generations += 1
        print("Evaluaciones realizadas: ", modified.sum())
        print("Vector de modificados: ", modified)

        # Ordenar la nueva poblacion por fitness
        new_pop_fitness, new_population = genetics.sort_population(new_pop_fitness, new_population)
        
        # Elitismo y sustitucion generacional de la poblacion
        pop_fitness, population = genetics.elitism(pop_fitness, population, new_pop_fitness, new_population)

        # Aplicar busqueda local sobre la poblacion cada 10 generaciones
        # Decidir sobre que parte de la poblacion aplicarla y actualizar
        # el numero de evaluaciones totales realizadas
        if n_generations % 10 == 0:
            if ls_rate == 1.0:
            		# Busqueda local sobre toda la poblacion
                print("Aplicando LS sobre toda la poblacion")
                for i in range(chromosomes):
                    # Ir modificando cada elemento de la poblacion con su fitness
                    population[i], pop_fitness[i] = local_search(data, labels, population[i], pop_fitness[i], ls_evals)
                    n_evaluations += ls_evals
            else:
                # Obtener el numero de elementos sobre los que realizar la busqueda local
                ls_elements = int(ls_rate * chromosomes)

                if ls_best:
                    # Aplicar la busqueda local sobre los ls_elements mejores cromosomas
                    print("Aplicando LS sobre los mejores ", ls_elements)
                    for i in range(ls_elements):
                        # Ir modificando el cromosoma y el fitness de los mejores de la poblacion
                        population[i], pop_fitness[i] = local_search(data, labels, population[i], pop_fitness[i], ls_evals)
                        n_evaluations += ls_evals
                else:
                    # Generar ls_elements indices aleatorios sobre los que aplicar la busqueda local
                    ls_index = np.random.choice(chromosomes, ls_elements, replace=True)

                    print("Aplicando LS sobre aleatorios ", ls_index)
                    for index in ls_index:
                        # Ir modificando los valores del cromosoma y del fitness  de los elementos aleatorios
                        population[index], pop_fitness[index] = local_search(data, labels, population[index], pop_fitness[index], ls_evals)
                        n_evaluations += ls_evals
            
            # Ordenar de nuevo la poblacion en caso de que se haya visto desordenada
            pop_fitness, population = genetics.sort_population(pop_fitness, population)


        print(population)
        print(pop_fitness)

    return population[0]
