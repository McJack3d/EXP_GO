import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

###GPT line to remove warnings
pd.set_option('future.no_silent_downcasting', True)

df0 = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP/BICLASS - Fitness/fitness_dataset.csv")

df0['sleep_hours'] = df0['sleep_hours'].fillna(df0['sleep_hours'].mean())
df0['smokes'] = df0['smokes'].dropna()

###Viz
#proportions = df0['is_fit'].value_counts(normalize = True)

#plt.scatter(df0['is_fit'].value_counts(normalize=True), df0['age'].value_counts())
#plt.show()

#df0.hist(bins=50, figsize=(12, 8))
#plt.title("Distrib features")
#plt.show()

#df0.hist(bins=30, figsize=(12, 8))
#plt.tight_layout()
#plt.show()

#print(df0.head())
#print(df0['height_cm'].nunique())
#spectrum = df0['age'].max() - df0['age'].min()
#print(spectrum)

###Cor

#df1 = df0.select_dtypes(include='number').corr()

#sns.heatmap(df1, annot=True, cmap='coolwarm')
#plt.show()

### Standardisation
df0['smokes'] = df0['smokes'].replace({'no': 0, 'yes': 1}).astype(int)
df0['gender'] = df0['gender'].replace({'F': 0, 'M': 1}).astype(int)

train, temp = train_test_split(df0, test_size=0.3, stratify=df0['is_fit'], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp['is_fit'], random_state=42)

#print(len(train), len(val), len(test))

####Baseline GPT
# Séparation features/target pour le jeu de test
#X_test = test.drop('is_fit', axis=1)
#y_test = test['is_fit']

# DummyClassifier baseline
#baseline = DummyClassifier(strategy='most_frequent')
#baseline.fit(X_test, y_test)
#y_pred = baseline.predict(X_test)

# Évaluation
#print('Accuracy:', accuracy_score(y_test, y_pred))
#print('F1-score:', f1_score(y_test, y_pred))

###Reg Logi
model = LogisticRegression(random_state=42)