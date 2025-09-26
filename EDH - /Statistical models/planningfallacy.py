import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import stats

df = pd.read_excel("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Statistical models/Planning_fallacy_Wooclap.xlsx")

mean = df["Planning"].mean()
variance= df["Planning"].var()

stats.probplot(df["Planning"],plot=plt)
plt.show()

