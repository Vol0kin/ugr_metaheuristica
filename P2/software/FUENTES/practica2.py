import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score                  # Medir la precision de los resultados de test
from sklearn.neighbors import KNeighborsClassifier          # Clasificador KNN
import time                                                 # Medir el tiempo
import sys                                                  # Argumentos de la linea de comandos

try:
    from pykdtree.kdtree import KDTree                      # Implementacion paralela de KDTree
except ImportError:
    from scipy.spatial import cKDTree as KDTree             # En caso de fallar, importar cKDTree como KDTree
