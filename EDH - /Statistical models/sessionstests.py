
import numpy as np
import pandas as pd
import matplotlib as plt
import scipy.stats as stats
"""
df = pd.read_excel("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Statistical models/Google_returns.xlsx")

mean = df["Daily Return"].mean()
variance= df["Daily Return"].var()

print(f"The average daily return of the Google stock is {mean*100:.2f}% and the variance is {variance}")

plt 

stats.binom.cdf(#successes,n=population,p=probability)

p = stats.binom.cdf(23, n= 37)
"""

df = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Statistical models/bank.csv")

y_count = df["y"]
y_mean = 521/4521
y_variance = y_mean * (1-y_mean)

print(f"Proportion: {y_mean:.4f}")
print(f"Percentage: {y_mean*100:.2f}%") 
print(f"Variance: {y_variance:.4f}")
print(df["y"])

