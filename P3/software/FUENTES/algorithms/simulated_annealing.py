import numpy as np
from . import metrics
from . import utils


def generate_t_start(C, mu=0.3, phi=0.3):
    """
    Funcion para generar la temperatura inicial segun el esquema
    modificado de Cauchy

    :param C: Coste de la solucion inicial
    :param mu: Valor que modifica el coste (por defecto 0.3)
    :param phi: Valor que divide el producto (por defecto 0.3)

    :return Devuelve la temperatura inicial calculada segun el esquema
            modificado de Cauchy
    """
    return (mu * C) / -np.log(phi)


def generate_beta(t_start, t_end, M):
    """
    Funcion para generar el beta utilizado en el enfriamiento mediante
    el esquema modificado de Cauchy

    :param t_start: Temperatura inicial
    :param t_end: Tempertarua final
    :param M: Numero de enfriamientos a realizar segun el esquema modificado
              de Cauchy

    :return Devuelve el valor de beta
    """

    return (t_start - t_end) / (M * t_start * t_end)


def cooling_scheme(temperature, beta):
    """
    Esquema de enfriamiento de Cauchy modificado. Reduce la temperatura

    :param temperature: Temmperatura que se va a reducir
    :param beta: Valor beta con el que se va a modificar la temperatura

    :return Devuelve la nueva temperatura calculada segun el esquema modificado
            de Cauchy
    """

    new_temmperature = temperature / (1 + beta * temperature)
    
    return new_temmperature


def neighbor_operator(w, mean=0.0, sigma=0.3):
    """
    Operador de movimiento por el vecindario. Genera un nuevo vecino

    :param w: Vector de pesos
    :param mu: Media del valor normal (por defecto 0.0)
    :param sigma: Desviacion tipica de la normal (por defecto 0.3)

    :return Devuelve un nuevo vector de pesos w del vecindario
    """
    
    # Generar vecino
    trait = np.random.choice(np.arange(w.shape[0]))
    neighbor = np.copy(w)
    neighbor[trait] += np.random.normal(mean, sigma)

    # Normalizar vecino
    neighbor = utils.normalize_w(neighbor)

    return neighbor



def simulated_annealing(X, y, max_evaluations=15000):
    """
    Funcion que realiza la busqueda basada en el enfriamiento simulado

    :param X: Conjunto de datos
    :param y: Conjunto de etiquetas
    :param max_evaluations: Numero maximo de evaluaciones que debe realizar
           el enfriamiento simulado (por defecto 15000)

    :return Devuelve un vector de pesos w resultado de haber realizado el
            enfriamiento simulado
    """

    # Establecer temperatura final
    t_end = 10**-3

    # Obtener numero de caracteristicas e inicializar w inicial y mejor w
    N = X.shape[1]
    w = utils.generate_initial_w(N)
    best_w = np.copy(w)

    # Establecer valores de max vecions, max exitos y M
    max_neigh = 10 * N
    max_success = round(0.1 * max_neigh)
    M = round(max_evaluations / max_neigh)

    # Inicializar evaluaciones, iteraciones y exitos
    num_evaluations = 0
    num_iterations = 0
    num_success = 1

    # Evaluar la solucion inicial y guardar como mejor evaluacion
    C = metrics.evaluate(X, y, w)
    best_fit = C
    current_fit = C
    num_evaluations += 1

    # Generar la temperatura inicial
    temperature = generate_t_start(C)

    # Generar beta
    beta = generate_beta(temperature, t_end, M)

    #print('Initial Temperature: ', temperature)
    #print('Final Temperature: ', t_end)
    #print('M: ', M)
    #print('N: ', N)
    #print('Max neighbors: ', max_neigh)
    #print('Max success: ', max_success)
    #print('Beta: ', beta)
    #print('w: ', best_w)

    # Mientras no se hayan dado M iteraciones y haya habido al menos un exito
    # realizar la busqueda mediante enfriamiento simulado
    while num_iterations < M and num_success > 0:
        # Inicializar numero de exitos y de vecinos actuales
        num_success = 0
        num_neigh = 0

        # Mientras no se hayan generado todos los vecinos y no se hayan
        # alcanzado el numero maximo de exitos, seguir generando vecinos
        while num_neigh < max_neigh and num_success < max_success:
            # Generar y evaluar nueva solucion
            new_w = neighbor_operator(w)
            new_fit = metrics.evaluate(X, y, new_w)
            num_evaluations += 1

            # Calcular delta
            delta_fit = new_fit - current_fit

            # Si se obtiene una mejor solucion o se genera un valor aleatorio
            # valido, actualizar solucion actual
            if delta_fit > 0 or np.random.uniform(0.0, 1.0) <= np.exp(delta_fit / temperature):
                num_success += 1
                w = new_w
                current_fit = new_fit

                # Actualizar mejor solucion en caso de mejorar
                if current_fit > best_fit:
                    best_w = w
                    best_fit = current_fit

            num_neigh += 1

        # Reducir temperatura
        temperature = cooling_scheme(temperature, beta)

        num_iterations += 1

    return best_w
