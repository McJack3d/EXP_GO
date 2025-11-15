import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

df = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Data Mining/HW1/fraud_homework_features.csv")

df_processed = df.copy()

positive_cols = ['Transaction_Amount', 'Items_in_Cart', 'IP_Reputation_Score']
df_processed[positive_cols] = np.log1p(df[positive_cols])

min_val = df['Transaction_vs_Avg'].min()
df_processed['Transaction_vs_Avg'] = df['Transaction_vs_Avg'] - min_val + 1
df_processed['Transaction_vs_Avg'] = np.log1p(df_processed['Transaction_vs_Avg'])

X_transformed = df_processed.values

mu = np.mean(X_transformed, axis=0)

cov = np.cov(X_transformed, rowvar=False)

inv_cov = np.linalg.inv(cov)

diff = X_transformed - mu
md_squared = np.sum(diff @ inv_cov * diff, axis=1)

K = X_transformed.shape[1]
chi2_threshold = chi2.ppf(0.999, df=K)

outliers = md_squared > chi2_threshold
my_predictions = outliers.astype(int)

pd.DataFrame(my_predictions).to_csv('predictions.csv', index=False, header=False)

print(df.describe())

plt.figure(figsize=(16, 6))
plt.plot(range(len(md_squared)), md_squared, linewidth=0.5, alpha=0.7, color='steelblue')
plt.scatter(range(len(md_squared)), md_squared, c=my_predictions, cmap='coolwarm', s=5, alpha=0.8)
plt.axhline(y=chi2_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {chi2_threshold:.2f}')
plt.xlabel('Index Transaction', fontsize=12)
plt.ylabel('Distance Mahalanobis²', fontsize=12)
plt.title('Distance de Mahalanobis² pour chaque transaction', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('all_mahalanobis_values.png', dpi=300, bbox_inches='tight')
plt.show()