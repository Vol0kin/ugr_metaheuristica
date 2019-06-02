import numpy as np
from . import metrics
from . import utils

###############################################################################
#                              Mutacion ILS                                   #
###############################################################################

def mutate(w, N, mut_rate=0.1, mean=0.0, sigma=0.4):
    """
    Funcion para mutar un vector de pesos en la busqueda ILS

    :param w: Vector de pesos sobre el que aplicar la mutacion
    :param N: Numero de elementos del vector de pesos
    :param mut_rate: Ratio de elementos a mutar (por defecto 0.1)
    :param mean: Media de la mutacion (por defecto 0.0)
    :param sigma: Desviacion tipica de la mutacion (por defecto 0.4)

    :return Devuelve un vector de pesos mutado y normalizado
    """

    # Obtener numero de mutaciones
    n_mutations = round(N * 0.1)

    # Elegir caracteristicas a mutar
    traits = np.random.choice(np.arange(N), n_mutations, replace=False)

    # Clonar w
    mut_w = np.copy(w)

    # Mutar caracteristicas
    mut_w[traits] += np.random.normal(mean, sigma)

    # Normalizar los valores de w
    mut_w = utils.normalize_w(mut_w)

    return mut_w

###############################################################################
#                   Operador de vecino en la busqueda local                   #
###############################################################################

def neighbor_operator(w, gene, mean=0.0, sigma=0.3):
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

def local_search(data, labels, initial_w, max_evaluations=1000):
    """
    Funcion para el calculo de w mediante la busqueda local

    :param data: Conjunto de datos
    :param labels: Conjunto de caracteristicas
    :param initial_w: Valor inicial de w
    :param initial_fit: Valor fitness inicial
    :param max_evaluations: Maximo numero de evaluaciones de la funcion fitness
                            que puede hacer la busqueda (por defecto 1000)

    :return Devuelve el w obtenido despues del proceso de busqueda local junto
            con su valor fitness
    """

    # Obtener numero de caracteristicas
    N = data.shape[1]

    # Copiar w inicial y valor fitness inicial
    w = np.copy(initial_w)
    fitness_val = metrics.evaluate(data, labels, w)

    # Inicializar evaluaciones
    evaluations = 1

    # Mientras no se hayan superado el numero maximo de evaluaciones
    # intentar ajustar w
    while evaluations < max_evaluations:
        # Copiar w
        current_w = np.copy(w)

        # Intentar mutar cada caracteristica en el orden dado por la permutacion
        # hasta encontrar la primera mutacion que obtiene un mejor valor de fitness
        for trait in np.random.permutation(N):
            # Mutar el w_i con un valor de la normal con media mean y d.t. sigma
            w = neighbor_operator(w, trait)

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


def ils(X, y, max_evaluations=15000):
    """
    Funcion que implementa la busqueda ILS. Calcula un vector de pesos a partir
    de unos datos de entrada aplicando busquedas locales iterativas hasta
    conseguir el mejor w posible

    :param X: Conjunto de datos
    :param y: Conjunto de etiquetas
    :param max_evaluations: Numero maximo de evaluaciones de la busqueda
                            (por defecto 15000)

    :return Devuelve un vector de pesos obtenido por ILS
    """

    # Obtener numero de elementos
    N = X.shape[1]

    # Inicializar numero de evaluaciones y incremento de evaluaciones
    n_evaluations = 0
    eval_inc = 1000

    # Generar w inicial y aplicar busqueda local
    initial_w = utils.generate_initial_w(N)

    best_w, best_fit = local_search(X, y, initial_w)
    n_evaluations += eval_inc

    # Aplicar ILS hasta que se llegue al numero maximo de evaluaciones
    while n_evaluations < max_evaluations:
        # Copiar el mejor w
        new_w = np.copy(best_w)

        # Mutar el nuevo w
        new_w = mutate(new_w, N)

        # Aplicar busqueda local sobre el nuevo w
        new_w, new_fit = local_search(X, y, new_w)
        n_evaluations += eval_inc

        # Actualizar mejor w y fitness en caso de que los nuevos sean mejores
        if new_fit > best_fit:
            best_w = new_w
            best_fit = new_fit

    return best_w
