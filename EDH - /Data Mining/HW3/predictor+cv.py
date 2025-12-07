import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error

df_train = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Data Mining/HW3/health_train.csv")
df_test = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Data Mining/HW3/health_test_features.csv")

# --- Outlier trimming on target Annual_Premium (IQR rule) ---
Q1 = df_train["Annual_Premium"].quantile(0.25)
Q3 = df_train["Annual_Premium"].quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

mask = (df_train["Annual_Premium"] >= lower_fence) & (df_train["Annual_Premium"] <= upper_fence)
print("Keeping", mask.sum(), "observations out of", len(df_train))

df_train = df_train.loc[mask].reset_index(drop=True)

def tobacco_flag(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"y": 1.0, "yes": 1.0, "true": 1.0, "1": 1.0})
        .fillna(0.0)
    )

df_train["Tobacco_Flag"] = tobacco_flag(df_train["Tobacco_User"])
df_test["Tobacco_Flag"] = tobacco_flag(df_test["Tobacco_User"])

# 2) Squared terms
df_train["Age2"] = df_train["Age_Years"] ** 2
df_test["Age2"] = df_test["Age_Years"] ** 2

df_train["BMI2"] = df_train["Body_Mass_Index"] ** 2
df_test["BMI2"] = df_test["Body_Mass_Index"] ** 2

# 3) Interactions with smoking
df_train["Age_Tobacco"] = df_train["Age_Years"] * df_train["Tobacco_Flag"]
df_test["Age_Tobacco"] = df_test["Age_Years"] * df_test["Tobacco_Flag"]

df_train["BMI_Tobacco"] = df_train["Body_Mass_Index"] * df_train["Tobacco_Flag"]
df_test["BMI_Tobacco"] = df_test["Body_Mass_Index"] * df_test["Tobacco_Flag"]


# Step 1: Separate numerical and categorical columns
categorical_columns = df_train.select_dtypes(include=['object']).columns.tolist()
numerical_columns = df_train.select_dtypes(include=['number']).columns.tolist()
feature_numerical_columns = [col for col in numerical_columns if col != 'Annual_Premium']

# Step 2: Standardize numerical variables
scaler = StandardScaler()
df_train_numerical_standardized = pd.DataFrame(
    scaler.fit_transform(df_train[feature_numerical_columns]),
    columns=feature_numerical_columns
)

# Step 3: One-hot encode categorical variables
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df_train[categorical_columns])
one_hot_df_train = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Step 4: Combine standardized numerical and encoded categorical data
df_train_final = pd.concat(
    [df_train_numerical_standardized.reset_index(drop=True),
     one_hot_df_train.reset_index(drop=True)],
    axis=1
)

y = df_train['Annual_Premium']
X = df_train_final

# -------------------------
# 5. Cross-validation (k-fold) with RMSE
#    Compare: (a) standard target vs (b) log-transformed target
# -------------------------

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# We will reuse the same KFold structure for both models
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# ----- (a) Standard linear regression on original target -----
lin_reg_std = LinearRegression()

rmse_scorer = make_scorer(rmse, greater_is_better=False)

cv_scores_std = cross_val_score(
    lin_reg_std,
    X,
    y,
    cv=kf,
    scoring=rmse_scorer
)

mean_rmse_std = -cv_scores_std.mean()
print("Standard model - CV RMSE scores (negative):", cv_scores_std)
print("Standard model - Mean CV RMSE:", mean_rmse_std)

# ----- (b) Linear regression on log-transformed target -----
# We evaluate RMSE on the original scale by exponentiating predictions
log_rmses = []

for train_idx, val_idx in kf.split(X):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    # Fit on log of the target
    lin_reg_log = LinearRegression()
    lin_reg_log.fit(X_train_fold, np.log(y_train_fold))

    # Predict on validation fold and bring back to original scale
    y_val_pred_log = lin_reg_log.predict(X_val_fold)
    y_val_pred = np.exp(y_val_pred_log)

    log_rmses.append(rmse(y_val_fold, y_val_pred))

mean_rmse_log = np.mean(log_rmses)
print("Log-target model - CV RMSE scores:", log_rmses)
print("Log-target model - Mean CV RMSE:", mean_rmse_log)

# -------------------------
# 6. Choose best model based on CV RMSE
# -------------------------

use_log_model = mean_rmse_log < mean_rmse_std
if use_log_model:
    print(">>> Using LOG-TARGET model for final training and predictions.")
    final_model = LinearRegression()
    final_model.fit(X, np.log(y))
else:
    print(">>> Using STANDARD model for final training and predictions.")
    final_model = LinearRegression()
    final_model.fit(X, y)

# -------------------------
# 7. Predict on df_test set
# -------------------------

# Apply same transformations to df_test set
df_test_numerical_standardized = pd.DataFrame(
    scaler.transform(df_test[feature_numerical_columns]),
    columns=feature_numerical_columns
)
one_hot_encoded_test = encoder.transform(df_test[categorical_columns])
one_hot_df_test = pd.DataFrame(one_hot_encoded_test, columns=encoder.get_feature_names_out(categorical_columns))

df_test_final = pd.concat(
    [df_test_numerical_standardized.reset_index(drop=True),
     one_hot_df_test.reset_index(drop=True)],
    axis=1
)

# Remove Annual_Premium column if it exists in df_test_final (it shouldn't, but just in case)
if 'Annual_Premium' in df_test_final.columns:
    X_test = df_test_final.drop('Annual_Premium', axis=1)
else:
    X_test = df_test_final

# Predict with the chosen final model
if use_log_model:
    test_predictions_log = final_model.predict(X_test)
    test_predictions = np.exp(test_predictions_log)
else:
    test_predictions = final_model.predict(X_test)

# Ensure it's a 1D array
test_predictions = np.asarray(test_predictions).reshape(-1)

# -------------------------
# 9. Export predictions.csv
# -------------------------

df_to_submit = pd.DataFrame(test_predictions)
df_to_submit.to_csv("predictions.csv", header=False, index=False)
print("predictions.csv saved!")