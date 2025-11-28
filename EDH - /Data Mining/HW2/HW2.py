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

print(df_std.describe())