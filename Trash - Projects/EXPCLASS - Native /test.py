import numpy as np
import sklearn as skl
import pandas as pd
import time
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

""" # data = load_wine()
df_wine = pd.DataFrame(data.data, columns=data.feature_names)
print(df_wine.head())

for col in data.feature_names:
     print(df_wine[col])
     print("Mean:", df_wine[col].mean())
     print("Median:", df_wine[col].median())
     print("Standard Deviation:", df_wine[col].std())

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B
print(C)

plt.scatter(C[0], C[1])
plt.show()

d = [np.random.exponential(100, 10) for i in range(10)]
plt.plot(d, label='Exponential Distribution')
plt.legend()
plt.show()

d = np.random.binomial(100, 0.1, 1000)
plt.plot(d, label='Binomial Distribution')
plt.legend()
plt.show()

print(pd.Series(d).describe())

m = 1000
x = np.arange(m)
trend = 3 ** x + 10
noise = np.random.normal(0,100,m)
y = trend + noise 

x = [2, 7, 9, 23, 28, 34, 42] 
y = [1, 3, 5, 7, 9, 11, 13]
plt.plot([i**3 for i in x], y)
plt.show()

print("Quel est votre nom: ")
x = input()
time.sleep(2)
print("Bonjour " + x)
time.sleep(1)
print("Quel est votre chiffre chance?")
y = input()
time.sleep(1)

total = len(y)
for i, val in enumerate(y, 1):
    percent = int((i / total) * 100)
    print(f"Chargement... {percent}% : {val}")
    time.sleep(1)  # pause d'1 seconde entre chaque affichage
print("Chargement terminé !")

time.sleep(1)

print("Calcul de votre chiffre chance...")
y_list = [1, 3, 5, 7, 9, 11, 13]
total = len(y_list)
for i, val in enumerate(y_list, 1):
    percent = int((i / total) * 100)
    print(f"Chargement... {percent}% : {val}")
    time.sleep(1)  # pause d'1 seconde entre chaque affichage
print("Chargement terminé !")

time.sleep(1)

print("Votre chiffre chance est " + str(y))
"""