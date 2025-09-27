import pandas as pd

df = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Python/Data/screenTime.csv")
df.set_index("user_id")

grouped_catocup = df.groupby(by="occupation")
mean_screentime = df["screen_time_hours"].mean()

print(grouped_catocup.head())