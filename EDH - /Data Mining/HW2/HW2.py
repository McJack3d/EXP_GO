import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sc
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

df = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Data Mining/HW2/Cancer_gene_PCA.csv", index_col=0)

#standardization (z-score)
df_std = (df - df.mean()) / df.std()
#conversion array numpy
X_std = df_std.values

#calcul de toutes les distances euclidiennes par paires entre les observations (en 1D)
prox_matrix = squareform(pdist(X_std, metric="euclidean"))

# Calcul de la matrice de liaison (linkage matrix) pour le clustering hiérarchique,
# en utilisant la méthode "single" et la distance euclidienne sur les données standardisées
linkage_matrix = sc.linkage(X_std, method="single", metric="euclidean")

from scipy.cluster.hierarchy import fcluster

K = 5 # choisis un nombre de clusters (tu pourras le changer / tester)

# attribution d'un numéro de cluster à chaque observation
labels_hc = fcluster(linkage_matrix, t=K, criterion="maxclust")

# mettre les labels dans un DataFrame et sauvegarder
pred = pd.DataFrame(labels_hc, index=df.index)
pred.to_csv("predictions_hc.csv", header=False)

from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster

for K in range(2, 11):  # tu peux changer la plage
    labels_hc = fcluster(linkage_matrix, t=K, criterion="maxclust")
    sil_score = silhouette_score(X_std, labels_hc, metric="euclidean")
    print("K =", K, " -> silhouette moyenne =", sil_score)

# ============================================================
# K-means clustering : comparaison des K via la silhouette
# ============================================================

from sklearn.cluster import KMeans

print("\nK-means : silhouette moyenne pour K = 2,...,10")
for K in range(2, 11):
    # initialisation de K-means
    kmeans = KMeans(n_clusters=K, n_init=20, random_state=0)
    # apprentissage + obtention des labels
    labels_km = kmeans.fit_predict(X_std)
    # calcul de la silhouette
    sil_score_km = silhouette_score(X_std, labels_km, metric="euclidean")
    print("K =", K, "-> silhouette moyenne (K-means) =", sil_score_km)

# Exemple : si on veut garder K = 2 pour K-means
K = 2
kmeans_final = KMeans(n_clusters=K, n_init=20, random_state=0)
labels_km_final = kmeans_final.fit_predict(X_std)

pred_km = pd.DataFrame(labels_km_final, index=df.index)
pred_km.to_csv("predictions_kmeans.csv", header=False)