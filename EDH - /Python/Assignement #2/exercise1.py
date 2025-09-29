import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Python/Data/global_climate_events_economic_impact_2020_2025.csv")

#df.groupby("year")["economic_impact_million_usd"].sum().plot(kind='bar')

#df.groupby("country")["economic_impact_million_usd"].sum().plot(kind='bar')

plt.scatter(df["severity"], df["economic_impact_million_usd"])

plt.show()

def estimate_regression(df, dependent_variable, independents, include_intercept):
    dependent_variable = str()