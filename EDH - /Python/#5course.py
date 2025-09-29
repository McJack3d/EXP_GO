import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Python/Data/screenTime.csv")
df = df.set_index("user_id")

grouped_mean = df.groupby("occupation")["screen_time_hours"].mean()

group_median = df.groupby("occupation")["screen_time_hours"].median()

independent_variable = 0

def regress_screen_time_on(independent_variable):
    independent_variable = df[screen_time_hours]
    mod = smf.ols(formula = "screen_time_hours", data=df)
    res = mod.fit()
    print(res.summary())

print(regress_screen_time_on(independent_variable))