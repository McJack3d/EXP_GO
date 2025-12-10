import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

df_train = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Data Mining/HW3/health_train.csv")
df_test = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Data Mining/HW3/health_test_features.csv")

def tobacco_flag(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"y": 1.0, "yes": 1.0, "true": 1.0, "1": 1.0})
        .fillna(0.0)
    )

for df in (df_train, df_test):
    df["Tobacco_Flag"] = tobacco_flag(df["Tobacco_User"])
    df["Age2"] = df["Age_Years"] ** 2
    df["BMI2"] = df["Body_Mass_Index"] ** 2
    df["Age_Tobacco"] = df["Age_Years"] * df["Tobacco_Flag"]
    df["BMI_Tobacco"] = df["Body_Mass_Index"] * df["Tobacco_Flag"]

y = df_train["Annual_Premium"]
X = df_train.drop(columns=["Annual_Premium"])

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()

preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numerical_columns),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical_columns),
])

model = Pipeline([
    ("prep", preprocess),
    ("gbr", GradientBoostingRegressor(random_state=42)),
])

model.fit(X, y)
test_preds = model.predict(df_test)
pd.DataFrame(test_preds).to_csv("predictions.csv", header=False, index=False)