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