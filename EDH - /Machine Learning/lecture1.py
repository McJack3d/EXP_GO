import mglearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_excel("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Machine Learning/boston_housing_preprocessed_dataset.xlsx")

print(df.head())

X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

lr = LinearRegression().fit(X_train, y_train)

#higher intercept is less good ? 

