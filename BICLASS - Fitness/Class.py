import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/BICLASS - Fitness/fitness_dataset.csv")

df["sleep_hours"].fillna(df["sleep_hours"].mean())



print(df["sleep_hours"].info())

