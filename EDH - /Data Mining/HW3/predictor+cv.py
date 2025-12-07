import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error

df_train = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Data Mining/HW3/health_train.csv")
df_test = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Data Mining/HW3/health_test_features.csv")

# Step 1: Separate numerical and categorical columns
categorical_columns = df_train.select_dtypes(include=['object']).columns.tolist()
numerical_columns = df_train.select_dtypes(include=['number']).columns.tolist()

# Step 2: Standardize numerical variables
scaler = StandardScaler()
df_train_numerical_standardized = pd.DataFrame(scaler.fit_transform(df_train[numerical_columns]), columns=numerical_columns)

# Step 3: One-hot encode categorical variables
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df_train[categorical_columns])
one_hot_df_train = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Step 4: Combine standardized numerical and encoded categorical data
df_train_final = pd.concat([df_train_numerical_standardized.reset_index(drop=True), one_hot_df_train.reset_index(drop=True)], axis=1)

# Prepare X and y (remove Annual_Premium from features)
y = df_train_final['Annual_Premium']
X = df_train_final.drop('Annual_Premium', axis=1)

# -------------------------
# 5. Cross-validation (k-fold) with RMSE
# -------------------------

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

lin_reg = LinearRegression()

kf = KFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    lin_reg,
    X,
    y,
    cv=kf,
    scoring=rmse_scorer
)

print("CV RMSE scores (negative):", cv_scores)
print("Mean CV RMSE:", -cv_scores.mean())

# -------------------------
# 7. Fit final model on full training set
# -------------------------

lin_reg.fit(X, y)

# -------------------------
# 8. Predict on test set
# -------------------------

# Apply same transformations to test set
df_test_numerical_standardized = pd.DataFrame(scaler.transform(df_test[numerical_columns]), columns=numerical_columns)
one_hot_encoded_test = encoder.transform(df_test[categorical_columns])
one_hot_df_test = pd.DataFrame(one_hot_encoded_test, columns=encoder.get_feature_names_out(categorical_columns))

df_test_final = pd.concat([df_test_numerical_standardized.reset_index(drop=True), one_hot_df_test.reset_index(drop=True)], axis=1)

# Remove Annual_Premium column if it exists in test set (it shouldn't, but just in case)
if 'Annual_Premium' in df_test_final.columns:
    X_test = df_test_final.drop('Annual_Premium', axis=1)
else:
    X_test = df_test_final

test_predictions = lin_reg.predict(X_test)

# Ensure it's a 1D array
test_predictions = np.asarray(test_predictions).reshape(-1)

# -------------------------
# 9. Export predictions.csv
# -------------------------

df_to_submit = pd.DataFrame(test_predictions)
df_to_submit.to_csv("predictions.csv", header=False, index=False)
print("predictions.csv saved!")