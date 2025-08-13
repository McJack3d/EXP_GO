import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as Kne

###GPT line to remove warnings
pd.set_option('future.no_silent_downcasting', True)

df0 = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP/BICLASS - Fitness/fitness_dataset.csv")

df0['sleep_hours'] = df0['sleep_hours'].fillna(df0['sleep_hours'].mean())
df0['smokes'] = df0['smokes'].dropna()

####TT le code suivant pour explorer le dataset
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
modelLR = LogisticRegression(random_state=42, max_iter=1000)

XLR_train = train.drop('is_fit', axis=1)
YLR_train = train['is_fit']
XLR_test = test.drop('is_fit', axis= 1)
YLR_test = test['is_fit']

###Hyperparameter tuning
param_grid = {'C' : [0.1, 1, 10], 'penalty' : ['l2']}

grid_search = GridSearchCV(modelLR, param_grid, cv=5, scoring='accuracy')
grid_search.fit(XLR_train, YLR_train)

modelLR.fit(XLR_train, YLR_train)
YLR_pred = modelLR.predict(XLR_test)
accuracy_score(YLR_test, YLR_pred), f1_score(YLR_test, YLR_pred)
# Print the accuracy and F1 score
print(f"AccuracyLR: {accuracy_score(YLR_test, YLR_pred)}, F1LR: {f1_score(YLR_test, YLR_pred)}")

###Random Forest
modelRF = RandomForestClassifier(random_state=42)

XRF_train = train.drop('is_fit', axis=1)
YRF_train = train['is_fit']
XRF_test = test.drop('is_fit', axis= 1)
YRF_test = test['is_fit']

modelRF.fit(XRF_train, YRF_train)
YRF_pred = modelRF.predict(XRF_test)
accuracy_score(YRF_test, YRF_pred), f1_score(YRF_test, YRF_pred)
# Print the accuracy and F1 score
print(f"AccuracyRF: {accuracy_score(YRF_test, YRF_pred)}, F1RF: {f1_score(YRF_test, YRF_pred)}")

###Knearestn
modelKN = Kne(n_neighbors=5)
XKN_train = train.drop('is_fit', axis=1)
YKN_train = train['is_fit']
XKN_test = test.drop('is_fit', axis= 1)
YKN_test = test['is_fit']

modelKN.fit(XKN_train, YKN_train)
YKN_pred = modelKN.predict(XKN_test)
accuracy_score(YKN_test, YKN_pred), f1_score(YKN_test, YKN_pred)

###Eval Finale
# Print the accuracy and F1 score
print(f"AccuracyKN: {accuracy_score(YKN_test, YKN_pred)}, F1KN: {f1_score(YKN_test, YKN_pred)}")

# Régression logistique
print("Logistic Regression")
print(classification_report(YLR_test, YLR_pred))
print(confusion_matrix(YLR_test, YLR_pred))

# Random Forest
print("Random Forest")
print(classification_report(YRF_test, YRF_pred))
print(confusion_matrix(YRF_test, YRF_pred))

# K-nearest neighbors 
print("K-Nearest Neighbors")
print(classification_report(YKN_test, YKN_pred))
print(confusion_matrix(YKN_test, YKN_pred))

# Importance des variables pour la forêt aléatoire
importances = modelRF.feature_importances_
features = XRF_train.columns
for feat, imp in zip(features, importances):
    print(f"{feat}: {imp}")

print("\n")
# Coefficients pour la régression logistique
coeffs = modelLR.coef_[0]
for feat, coef in zip(XLR_train.columns, coeffs):
    print(f"{feat}: {coef}")