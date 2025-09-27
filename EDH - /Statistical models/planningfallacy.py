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

