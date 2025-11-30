import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

FILEPATH = "Cancer_gene_PCA.csv"

df = pd.read_csv(FILEPATH, index_col=0)
variances = df.var()
sorted_columns = variances.sort_values(ascending=False).index
df_reduced = df[sorted_columns].iloc[:, :2]

k = 4
Z = sch.linkage(df_reduced, method='ward', metric='euclidean')
labels = sch.fcluster(Z, t=k, criterion='maxclust')

labels = labels - 1

pd.DataFrame(labels).to_csv('predictions.csv', header=False, index=False)