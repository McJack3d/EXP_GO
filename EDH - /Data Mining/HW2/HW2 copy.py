import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILEPATH = "/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Data Mining/HW2/Cancer_gene_PCA.csv"

# ==========================================
# 1. LOAD DATA
# ==========================================
print("--- Step 1: Loading Data ---")

df = pd.read_csv(FILEPATH, index_col=0)
variances = df.var()
sorted_columns = variances.sort_values(ascending=False).index
df_sorted = df[sorted_columns]

print(f"Total samples: {len(df)}")
print(f"Total features: {len(sorted_columns)}")

# ==========================================
# 2. COMPUTE VARIANCE EXPLAINED
# ==========================================
print("\n--- Step 2: Computing Variance Explained ---")

# Total variance = 20,531 (as per HW sheet)
TOTAL_VARIANCE = 20531

# Variance of each PC (already sorted)
pc_variances = variances[sorted_columns].values

# Explained variance ratio
explained_variance_ratio = pc_variances / TOTAL_VARIANCE
cumulative_variance = np.cumsum(explained_variance_ratio)

# Key thresholds
n_70 = np.argmax(cumulative_variance >= 0.70) + 1
n_80 = np.argmax(cumulative_variance >= 0.80) + 1
n_90 = np.argmax(cumulative_variance >= 0.90) + 1

print(f"PCs for 70% variance: {n_70}")
print(f"PCs for 80% variance: {n_80}")
print(f"PCs for 90% variance: {n_90}")

# ==========================================
# 3. SCREE PLOT
# ==========================================
print("\n--- Step 3: Generating Scree Plot ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Individual Variance (Eigenvalues)
ax = axes[0]
ax.bar(range(1, 51), explained_variance_ratio[:50] * 100, color='steelblue', edgecolor='black')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained (%)')
ax.set_title('Scree Plot - Individual Variance (Top 50 PCs)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Cumulative Variance
ax = axes[1]
ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, 'b-', linewidth=2)
ax.axhline(y=70, color='green', linestyle='--', linewidth=1, label='70%')
ax.axhline(y=80, color='orange', linestyle='--', linewidth=1, label='80%')
ax.axhline(y=90, color='red', linestyle='--', linewidth=1, label='90%')
ax.axvline(x=n_70, color='green', linestyle=':', linewidth=1)
ax.axvline(x=n_80, color='orange', linestyle=':', linewidth=1)
ax.axvline(x=n_90, color='red', linestyle=':', linewidth=1)
ax.scatter([n_70, n_80, n_90], [70, 80, 90], c=['green', 'orange', 'red'], s=100, zorder=5)
ax.annotate(f'n={n_70}', (n_70, 70), textcoords="offset points", xytext=(10, -10))
ax.annotate(f'n={n_80}', (n_80, 80), textcoords="offset points", xytext=(10, -10))
ax.annotate(f'n={n_90}', (n_90, 90), textcoords="offset points", xytext=(10, -10))
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Variance Explained (%)')
ax.set_title('Cumulative Variance Explained', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 200)

plt.suptitle('PCA Scree Plot Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('scree_plot.png', dpi=150)
plt.show()

# ==========================================
# 4. DETAILED TABLE (TOP 20 PCs)
# ==========================================
print("\n--- Step 4: Top 20 PCs Summary ---")

summary_df = pd.DataFrame({
    'PC': range(1, 21),
    'Variance': pc_variances[:20],
    'Variance %': explained_variance_ratio[:20] * 100,
    'Cumulative %': cumulative_variance[:20] * 100
})
print(summary_df.to_string(index=False))
