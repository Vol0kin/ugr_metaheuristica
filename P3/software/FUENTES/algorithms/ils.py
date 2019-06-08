import numpy as np
from . import utils
from .local_search import local_search

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
