import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Statistical models/Monthly_SP500_returns.xlsx")

df_corr = df.corr()

plt.plot(df_corr["NVDA"])
plt.show()

print(df_corr["NVDA"])

