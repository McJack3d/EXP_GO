"""
ML PRACTICAL EXAM – REVISION SCRIPT
-----------------------------------

Covers:
- Classification: Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, SVM
- Regression: Linear Regression, Ridge, Lasso
- Bias–variance through train vs test performance
- Cross-validation vs single train/test split
- Appropriate metrics for regression vs classification
- Tiny stump exercise like in the mock exam

Run this file section by section from VS Code.
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso


# =========================================================
# PART 0 – HELPER
# =========================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =========================================================
# PART 1 – CLASSIFICATION ON BREAST CANCER
# (Naive Bayes, Tree, RF, Gradient Boosting, SVM, CV)
# =========================================================

print_section("PART 1 – CLASSIFICATION ON BREAST CANCER")

# 1.1 Load data
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))
print("Class distribution:", np.bincount(y))

# 1.2 Train/test split (like in the mock exam)
# TODO: split data into train and test with test_size=0.3, random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 1.3 Define models in a dict
models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree (max_depth=3)": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
    ),
    "SVM (RBF kernel)": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
        ]
    ),
}

# 1.4 For each model: fit, compute train & test accuracy + 5-fold CV accuracy
results_cls = []

for name, model in models.items():
    print_section(f"Model: {name}")
    # TODO: fit the model on (X_train, y_train)
    model.fit(X_train, y_train)

    # TODO: compute train_accuracy and test_accuracy with accuracy_score
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # TODO: perform 5-fold cross_val_score with scoring='accuracy'
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    cv_mean = cv_scores.mean()

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test  accuracy: {test_acc:.3f}")
    print(f"5-fold CV accuracy (mean): {cv_mean:.3f}")
    print(f"5-fold CV scores: {cv_scores}")

    # store
    results_cls.append(
        {
            "model": name,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "cv_mean": cv_mean,
        }
    )

# 1.5 Inspect confusion matrix & report for the two best models (by test accuracy)
df_cls = pd.DataFrame(results_cls)
print_section("SUMMARY – CLASSIFICATION MODELS")
print(df_cls.sort_values("test_acc", ascending=False))

# TODO (by hand after printing):
# 1) Which model has the highest test accuracy?

# 2) Which model clearly overfits (train_acc >> test_acc)?
# 3) Which models seem to manage bias-variance best (train ≈ test ≈ CV)?


# pick the top two models by test accuracy
top2_names = df_cls.sort_values("test_acc", ascending=False)["model"].head(2).tolist()
print("Top 2 models:", top2_names)

for name in top2_names:
    print_section(f"Confusion matrix & report – {name}")
    model = models[name]
    y_pred = model.predict(X_test)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

# TODO: Compare confusion matrices and comment:
# - Does one model make more false positives / false negatives?
# - If this were a medical exam, which type of error is more problematic and why?


# =========================================================
# PART 2 – RANDOM FOREST VS GRADIENT BOOSTING
# (BIAS–VARIANCE & HYPERPARAMETERS)
# =========================================================

print_section("PART 2 – RANDOM FOREST VS GRADIENT BOOSTING (TUNING)")

# 2.1 Define baseline models again (simple versions)
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

# 2.2 Check how performance changes with n_estimators for each
n_estimators_list = [10, 50, 100, 300]

rf_scores = []
gb_scores = []

for n in n_estimators_list:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42)
    gb_temp = GradientBoostingClassifier(n_estimators=n, random_state=42)

    rf_cv = cross_val_score(rf_temp, X, y, cv=5, scoring="accuracy").mean()
    gb_cv = cross_val_score(gb_temp, X, y, cv=5, scoring="accuracy").mean()

    rf_scores.append(rf_cv)
    gb_scores.append(gb_cv)

    print(f"n_estimators={n} | RF CV acc={rf_cv:.3f} | GB CV acc={gb_cv:.3f}")

# TODO (by hand):
# - Do both methods improve when n_estimators increases?
# - Does one plateau earlier?
# - How does this relate to the way forests (bagging) vs boosting work conceptually?


# 2.3 GridSearchCV example for Gradient Boosting (like exam model selection)
param_grid_gb = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [2, 3],
}

gb_base = GradientBoostingClassifier(random_state=42)

# TODO: set up GridSearchCV with cv=5, scoring="accuracy"
gb_grid = GridSearchCV(
    estimator=gb_base,
    param_grid=param_grid_gb,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

# TODO: fit the grid search on the whole dataset (X, y)
gb_grid.fit(X, y)

print("Best GB params:", gb_grid.best_params_)
print("Best GB CV accuracy:", gb_grid.best_score_)

# TODO (conceptual, by hand):
# - Explain why using GridSearchCV with CV is more reliable than just a single
#   train/test split when comparing hyperparameter choices (mock exam Q4 style).


# =========================================================
# PART 3 – REGRESSION ON HOUSE PRICES (CALIFORNIA)
# (Linear, Polynomial, Ridge, Lasso, Metrics)
# =========================================================

print_section("PART 3 – REGRESSION ON CALIFORNIA HOUSING")

# 3.1 Load regression dataset (house prices analogue to exam question)
cal = fetch_california_housing()
X_reg = cal.data
y_reg = cal.target
feature_names_reg = cal.feature_names

print("Regression dataset shape:", X_reg.shape)

# 3.2 Train/test split
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# 3.3 Define three regression models:
#  - Linear Regression
#  - Ridge Regression
#  - Lasso Regression
# with scaling inside pipelines.

models_reg = {
    "Linear": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("linreg", LinearRegression()),
        ]
    ),
    "Ridge (alpha=1.0)": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=42)),
        ]
    ),
    "Lasso (alpha=0.01)": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=0.01, max_iter=10000, random_state=42)),
        ]
    ),
}

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

results_reg = []

for name, model in models_reg.items():
    print_section(f"Regression model: {name}")

    # TODO: fit on (Xr_train, yr_train)
    model.fit(Xr_train, yr_train)

    # TODO: compute train & test RMSE and R^2
    yhat_train = model.predict(Xr_train)
    yhat_test = model.predict(Xr_test)

    train_rmse = rmse(yr_train, yhat_train)
    test_rmse = rmse(yr_test, yhat_test)
    train_r2 = r2_score(yr_train, yhat_train)
    test_r2 = r2_score(yr_test, yhat_test)

    print(f"Train RMSE: {train_rmse:.3f}, Test RMSE: {test_rmse:.3f}")
    print(f"Train R^2 : {train_r2:.3f}, Test R^2 : {test_r2:.3f}")

    results_reg.append(
        {
            "model": name,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "test_r2": test_r2,
        }
    )

df_reg = pd.DataFrame(results_reg)
print_section("SUMMARY – REGRESSION MODELS")
print(df_reg.sort_values("test_rmse"))

# TODO (by hand):
# - Which model has the lowest test RMSE?
# - Which model has the highest test R^2?
# - Is there a model that clearly overfits (train ≪ test RMSE or train R^2 ≫ test R^2)?
# - Relate this to bias–variance and to ridge/lasso regularisation (coeff shrinkage).


# 3.4 (Optional, more advanced) – Polynomial features + Ridge
print_section("OPTIONAL – POLYNOMIAL FEATURES + RIDGE")

poly_ridge = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ]
)

poly_ridge.fit(Xr_train, yr_train)
yhat_train_pr = poly_ridge.predict(Xr_train)
yhat_test_pr = poly_ridge.predict(Xr_test)

print("Polynomial Ridge – Train RMSE:", rmse(yr_train, yhat_train_pr))
print("Polynomial Ridge – Test  RMSE:", rmse(yr_test, yhat_test_pr))

# TODO (by hand):
# - Does adding polynomial features help or hurt generalisation?
# - Connect this to the bias–variance trade-off.


# =========================================================
# PART 4 – MINI STUMP EXERCISE (DECISION TREE OF DEPTH 1)
# – EXAM STYLE
# =========================================================

print_section("PART 4 – MINI STUMP (DEPTH=1) ON A TOY DATASET")

# We'll build a tiny dataset similar in spirit to the mock exam’s tree question.
# Attendance: 0=Low, 1=High
# Hours: numeric
X_toy = np.array(
    [
        [0, 4],
        [1, 6],
        [0, 7],
        [1, 9],
        [1, 12],
        [0, 15],
    ]
)
y_toy = np.array([0, 1, 0, 1, 1, 0])

print("Toy X:\n", X_toy)
print("Toy y:", y_toy)

# TODO (BY HAND, WITHOUT CODE FIRST): 
# 1) Try to construct the best decision stump (depth=1) by eye:
#    - Either split on Attendance (0 vs 1)
#    - Or split on Hours (choose a threshold)
#    Compute Gini before split, Gini after split, and the gain for each candidate.
# 2) Choose the best stump and write it in words:
#    e.g. "If Attendance=High then predict 1, else 0".

# Now check your stump using sklearn with max_depth=1
stump = DecisionTreeClassifier(max_depth=1, criterion="gini", random_state=42)
stump.fit(X_toy, y_toy)

print("Sklearn stump – feature index used:", stump.tree_.feature[0])
print("Sklearn stump – threshold:", stump.tree_.threshold[0])

# feature index 0 = Attendance, 1 = Hours
# TODO: Compare sklearn’s stump with yours:
# - Same feature and threshold?
# - If different, compute the training accuracy of your stump vs sklearn’s.


print_section("END – You’ve completed the practical revision!")
print(
    "Now go back over your handwritten answers (TODOs) and make sure you "
    "can explain them *without* the code – that’s what you need for the exam."
)