import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sc
from scipy.spatial.distance import pdist, squareform

df = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Data Mining/HW2/Cancer_gene_PCA.csv")

print("="*60)
print("DONNÉES INITIALES")
print("="*60)
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nTypes de colonnes:")
print(df.dtypes.value_counts())

# Identifier la colonne non-numérique
print(f"\nColonnes non-numériques:")
non_numeric = df.select_dtypes(include=['object']).columns.tolist()
print(non_numeric)

# Séparer les labels (si présents) et les features numériques
if len(non_numeric) > 0:
    labels = df[non_numeric[0]]  # Sauvegarder les labels
    df_numeric = df.select_dtypes(include=[np.number])  # Garder seulement les colonnes numériques
    print(f"\nLabels sauvegardés: {non_numeric[0]}")
else:
    df_numeric = df
    labels = None

print(f"\nShape après sélection numérique: {df_numeric.shape}")
print(df_numeric.describe())

# ========== 1. STANDARDISATION DES PC ==========
print("\n" + "="*60)
print("STANDARDISATION DES PC")
print("="*60)

scaler = StandardScaler()
df_standardized = scaler.fit_transform(df_numeric)

# Convertir en DataFrame pour faciliter la visualisation
df_std = pd.DataFrame(df_standardized, columns=df_numeric.columns)

print("Après standardisation:")
print(f"Moyenne de chaque PC (doit être ~0):")
print(df_std.mean().head())
print(f"\nÉcart-type de chaque PC (doit être ~1):")
print(df_std.std().head())

# ========== 2. MATRICE DE PROXIMITÉ (Distance Euclidienne) ==========
print("\n" + "="*60)
print("MATRICE DE PROXIMITÉ")
print("="*60)

# Calculer la matrice de distance euclidienne sur les données standardisées
proximity_matrix = squareform(pdist(df_standardized, metric='euclidean'))

print(f"Shape de la matrice de proximité: {proximity_matrix.shape}")
print(f"Distance entre observation 0 et 1: {proximity_matrix[0, 1]:.6f}")
print(f"Distance entre observation 0 et 2: {proximity_matrix[0, 2]:.6f}")

# ========== 3. DENDROGRAMME - SINGLE LINKAGE ==========
print("\n" + "="*60)
print("CLUSTERING HIÉRARCHIQUE - SINGLE LINKAGE")
print("="*60)

# Créer le dendrogramme
plt.figure(figsize=(20, 7))
dendrogram = sc.dendrogram(sc.linkage(df_standardized, method='single', metric='euclidean'),
                          labels=labels.values if labels is not None else None)
plt.title('Dendrogramme - Single Linkage (Standardized Euclidean Distance)', 
         fontsize=14, fontweight='bold')
plt.xlabel('Index des observations', fontsize=12)
plt.ylabel('Distance euclidienne standardisée', fontsize=12)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('dendrogram_single_linkage.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Dendrogramme sauvegardé: 'dendrogram_single_linkage.png'")

# ========== 4. STATISTIQUES ==========
print("\n" + "="*60)
print("STATISTIQUES")
print("="*60)
print(f"Nombre d'observations: {len(df_numeric)}")
print(f"Nombre de features (PC): {df_numeric.shape[1]}")
print(f"Distance moyenne entre observations: {proximity_matrix[np.triu_indices_from(proximity_matrix, k=1)].mean():.4f}")
print(f"Distance min: {proximity_matrix[np.triu_indices_from(proximity_matrix, k=1)].min():.4f}")
print(f"Distance max: {proximity_matrix[np.triu_indices_from(proximity_matrix, k=1)].max():.4f}")