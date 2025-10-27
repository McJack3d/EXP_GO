import mglearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_excel("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Machine Learning/boston_housing_extended_dataset.xlsx")

X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

lr = LinearRegression().fit(X_train, y_train)

# scatter points
plt.scatter(X_train, y_train, label="data")
plt.scatter(X_test, y_test, label = "test", marker = '^') 

# plot linear regression line
X_line = np.linspace(-3, 3, 1000).reshape(-1, 1)
y_line = lr.predict(X_line)
plt.plot(X_line, y_line, color="red", linewidth=2, label="linear fit")

plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.tight_layout()
plt.show()