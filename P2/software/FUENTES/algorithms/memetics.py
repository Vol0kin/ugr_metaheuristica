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


def local_search(data, labels, initial_w, initial_fit, max_evaluations):
    """
    Funcion para el calculo de w mediante la búsqueda local.

    :param data: Conjunto de datos
    :param labels: Conjunto de caracteristicas
    :param initial_w: Valor inicial de w
    :param initial_fit: Valor fitness inicial
    :param max_evaluations: Maximo numero de evaluaciones de la funcion fitness
                            que puede hacer la busqueda

    :return Devuelve el w obtenido despues del proceso de busqueda local
    """

    # Obtener numero de caracteristicas
    N = data.shape[1]

    # Copiar w inicial y valor fitness inicial
    w = np.copy(initial_w)
    fitness_val = initial_fit

    # Inicializar evaluaciones
    evaluations = 0
    unsuccessful_evaluations = 0

    # Mientras no se hayan superado el numero maximo de evaluaciones
    # intentar ajustar w
    while evaluations < max_evaluations:
        # Copiar w
        current_w = np.copy(w)

        # Intentar mutar cada caracteristica en el orden dado por la permutacion
        # hasta encontrar la primera mutacion que obtiene un mejor valor de fitness
        for trait in np.random.permutation(N):
            # Mutar el w_i con un valor de la normal con media mean y d.t. sigma
            neighbor_mutation(w, trait)

            # Evaluar el nuevo w con los datos
            evaluations += 1
            new_fitness = metrics.evaluate(data, labels, w)

            # Si el nuevo valor de fitness es mejor, conservar w, guardar el mejor
            # valor de fitness e indicar que se ha realizado una evaluacion con 
            # exito
            if new_fitness > fitness_val:
                fitness_val = new_fitness
                unsuccessful_evaluations = 0
                break
            # En caso contrario, contabilizar un error mas y restaurar los w
            else:
                unsuccessful_evaluations += 1
                w[trait] = current_w[trait]

            # Si se sobrepasa el numero max de evaluaciones o se obtienen demasiados 
            # errores, se termina la busqueda
            if evaluations > max_evaluations or unsuccessful_evaluations > max_traits_evaluations * N:
                return w

    return w
