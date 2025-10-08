import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_excel("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Statistical models/life_gdp.xlsx")

x = df["GDPpcap"].values.reshape(-1, 1)
y = df["Life exp,"].values

model = LinearRegression()
model.fit(x, y)

y_fit = model.predict(x)

beta0 = model.intercept_
beta1 = model.coef_[0]

plt.scatter(df["GDPpcap"], df["Life exp,"])
plt.show()