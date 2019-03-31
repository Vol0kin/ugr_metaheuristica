import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_knn = pd.read_csv('out/ionosphere_knn_result.csv')
df_relief = pd.read_csv('out/ionosphere_relief_result.csv')
df_bl = pd.read_csv('out/ionosphere_local_result.csv')

data = np.array([[a, b, c] for a, b, c in zip(df_knn.values[:, 1], df_relief.values[:, 1], df_bl.values[:, 1])])
cols = ['1-NN', 'RELIEF', 'BL']
out_df = pd.DataFrame(data, columns=cols)

boxplot = out_df.boxplot(column=cols)

plt.title('Comparación de los algoritmos en el conjunto de datos Ionosphere.')
plt.ylabel('Tasa de clasificación')
plt.show()
