import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sc
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

df = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Data Mining/HW2/Cancer_gene_PCA.csv", index_col=0)

df_std = (df - df.mean()) / df.std()
X_std = df_std.values

prox_matrix = squareform(pdist(X_std, metric="euclidean"))

dist_1_2 = prox_matrix[0, 1]

linkage_matrix = sc.linkage(X_std, method="single", metric="euclidean")
"""
plt.figure(figsize=(12, 6))
sc.dendrogram(linkage_matrix)
plt.title("Dendrogram - Single linkage (standardized Euclidean)")
plt.xlabel("Observation index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()
"""
inertias = []
K_range = range(1, 501)  # k = 1..10

for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(X_std)
    inertias.append(kmeans.inertia_)  # SSE

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o')
plt.xlabel("Number of clusters k")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow graph (KMeans on standardized PCs)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()