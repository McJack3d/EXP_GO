import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import stats

df = pd.read_excel("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Statistical models/Planning_fallacy_Wooclap.xlsx")

mean = df["Planning"].mean()
variance= df["Planning"].var(ddof=1)

stats.probplot(df["Planning"],plot=plt)
#plt.show()

#After a brief review of the qqplot we can conclude that the variables follow a normal distribution

print(variance)

## Average is an estimate of the expectation
#The expectation of the "Planning" variable is 1.0188679245283019 (same value as the mean)

#The standard error of the estimator is squared root of (the empirical variance divided by the population N(sample size))

N = len(df) #Sample size

standard_error = np.sqrt(variance / N)
print(standard_error)

p_value = 1-stats.norm.cdf(mean, loc=0, scale=standard_error)
print(p_value)

#The p-value is way larger than 0.05. We do not reject the Null hypothesis that the expectation is equal to 0. We conclude that the sample does not provide evidence in favor of a planning fallacy in this context.