"""
Script ML amélioré pour classification binaire `is_fit`
- Jeu de données attendu avec colonnes:
  ['age','height_cm','weight_kg','heart_rate','blood_pressure','sleep_hours',
   'nutrition_quality','activity_index','smokes','gender','is_fit']

Principaux apports vs. version initiale:
1) Pipelines propres (imputation + scaling quand nécessaire) par modèle
2) GridSearchCV (5-fold stratifié) pour LR / RF / KNN
3) Baseline DummyClassifier
4) Evaluation unifiée (Accuracy, F1, rapport, matrice de confusion)
5) Interprétation: coefficients LR (standardisés + odds ratios),
   importances RF (impureté + permutation)
6) Séparation Train/Val/Test reproductible et claire

Comment utiliser:
- Place ce fichier à côté de ton notebook/script, puis importe et lance `main()`
  OU exécute directement en ligne de commande: `python this_file.py --csv path/to/data.csv`
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Pour rendre Pandas explicite sur certains changements
pd.set_option("future.no_silent_downcasting", True)


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_data(csv_path: str | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Charge un DataFrame depuis CSV ou utilise df s'il est fourni.
    Nettoyage minimal et typage cohérent.
    """
    if (csv_path is None) and (df is None):
        raise ValueError("Fournis soit csv_path, soit df (DataFrame déjà chargé).")
    data = pd.read_csv(csv_path) if df is None else df.copy()

    # Colonnes attendues
    expected = [
        'age','height_cm','weight_kg','heart_rate','blood_pressure',
        'sleep_hours','nutrition_quality','activity_index','smokes','gender','is_fit'
    ]
    missing = [c for c in expected if c not in data.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    # Harmonisation types pour binaires et cible
    data['smokes'] = (
        data['smokes']
        .replace({'no': 0, 'yes': 1, 'No': 0, 'Yes': 1, False: 0, True: 1})
    ) 
    data['smokes'] = pd.to_numeric(data['smokes'], errors='coerce')
    data['smokes'] = data['smokes'].astype('Int64')

    data['gender'] = (
        data['gender']
        .replace({'F': 0, 'M': 1, 'female': 0, 'male': 1, 'Female': 0, 'Male': 1})
    )
    data['gender'] = pd.to_numeric(data['gender'], errors='coerce')
    data['gender'] = data['gender'].astype('Int64')

    data['is_fit'] = (
        data['is_fit']
        .replace({False: 0, True: 1, 'no': 0, 'yes': 1})
    )
    data['is_fit'] = pd.to_numeric(data['is_fit'], errors='coerce')
    data['is_fit'] = data['is_fit'].astype('Int64')

    return data


def split_data(df0: pd.DataFrame) -> DataSplits:
    X = df0.drop(columns=['is_fit'])
    y = df0['is_fit'].astype(int)

    # Train/Temp (70/30)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    # Val/Test (15/15) via split 50/50 du temp (stratifié)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    # Recompose Train+(Val) pour la CV (conseillé si on fait du GridSearch)
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    return DataSplits(X_train_full, X_test, y_train_full, y_test)


def build_preprocessors(numeric_cols, binary_cols):
    """Deux preprocessors: avec et sans scaling pour les modèles sensibles à l'échelle."""
    # Imputation mediane pour numériques; mode pour binaires
    num_imp = SimpleImputer(strategy='median')
    bin_imp = SimpleImputer(strategy='most_frequent')

    # Avec scaling (pour LR & KNN)
    pre_with_scale = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imp", num_imp), ("scaler", StandardScaler())]), numeric_cols),
            ("bin", Pipeline(steps=[("imp", bin_imp)]), binary_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )
    # Sans scaling (pour RF)
    pre_no_scale = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imp", num_imp)]), numeric_cols),
            ("bin", Pipeline(steps=[("imp", bin_imp)]), binary_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )
    return pre_with_scale, pre_no_scale


def fit_models(splits: DataSplits, feature_cols: Dict[str, list]):
    X_train, X_test, y_train, y_test = splits.X_train, splits.X_test, splits.y_train, splits.y_test

    # Préprocesseurs
    pre_with_scale, pre_no_scale = build_preprocessors(
        numeric_cols=feature_cols['num'], binary_cols=feature_cols['bin']
    )

    # ===== Baseline =====
    dummy = Pipeline([
        ("pre", pre_no_scale),
        ("clf", DummyClassifier(strategy='most_frequent')),
    ])
    dummy.fit(X_train, y_train)

    # ===== Logistic Regression =====
    lr_pipe = Pipeline([
        ("pre", pre_with_scale),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE))
    ])
    lr_grid = {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__penalty": ["l2"],  # lbfgs -> L2
    }
    lr_cv = GridSearchCV(lr_pipe, lr_grid, cv=CV, scoring="f1", n_jobs=-1, refit=True)
    lr_cv.fit(X_train, y_train)

    # ===== Random Forest =====
    rf_pipe = Pipeline([
        ("pre", pre_no_scale),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    rf_grid = {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features": ["sqrt", "log2"],
    }
    rf_cv = GridSearchCV(rf_pipe, rf_grid, cv=CV, scoring="f1", n_jobs=-1, refit=True)
    rf_cv.fit(X_train, y_train)

    # ===== KNN =====
    knn_pipe = Pipeline([
        ("pre", pre_with_scale),
        ("clf", KNeighborsClassifier())
    ])
    knn_grid = {
        "clf__n_neighbors": [3, 5, 7, 9, 15],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],  # 1=Manhattan, 2=Euclidienne
    }
    knn_cv = GridSearchCV(knn_pipe, knn_grid, cv=CV, scoring="f1", n_jobs=-1, refit=True)
    knn_cv.fit(X_train, y_train)

    models = {
        "dummy": dummy,
        "lr": lr_cv.best_estimator_,
        "rf": rf_cv.best_estimator_,
        "knn": knn_cv.best_estimator_,
    }
    best_params = {
        "lr": lr_cv.best_params_,
        "rf": rf_cv.best_params_,
        "knn": knn_cv.best_params_,
    }
    return models, best_params


def evaluate_models(models: Dict[str, Pipeline], splits: DataSplits) -> pd.DataFrame:
    X_test, y_test = splits.X_test, splits.y_test
    rows = []
    for name, mdl in models.items():
        y_pred = mdl.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        rows.append({"model": name, "accuracy": acc, "f1": f1})
        print("\n===", name.upper(), "===")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def get_feature_names(preprocessor: ColumnTransformer, cols_num: list, cols_bin: list) -> np.ndarray:
    # Avec verbose_feature_names_out=False, get_feature_names_out retourne les noms d'origine après transformation
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback (scikit-learn <1.0)
        names = np.array(cols_num + cols_bin)
    return names


def interpret_lr(model_lr: Pipeline, feature_cols: Dict[str, list]):
    print("\n--- Interprétation Régression Logistique ---")
    pre: ColumnTransformer = model_lr.named_steps['pre']
    clf: LogisticRegression = model_lr.named_steps['clf']

    feat_names = get_feature_names(pre, feature_cols['num'], feature_cols['bin'])
    coefs = clf.coef_.ravel()
    odds = np.exp(coefs)
    dfc = pd.DataFrame({
        'feature': feat_names,
        'coef_standardized': coefs,
        'odds_ratio(exp(coef))': odds,
    }).sort_values('coef_standardized', key=np.abs, ascending=False)
    print(dfc.to_string(index=False))
    return dfc


def interpret_rf(model_rf: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, feature_cols: Dict[str, list]):
    print("\n--- Interprétation Random Forest ---")
    pre: ColumnTransformer = model_rf.named_steps['pre']
    clf: RandomForestClassifier = model_rf.named_steps['clf']
    feat_names = get_feature_names(pre, feature_cols['num'], feature_cols['bin'])

    # Importances par impureté
    imp = getattr(clf, 'feature_importances_', None)
    if imp is not None:
        df_imp = pd.DataFrame({'feature': feat_names, 'gini_importance': imp}) \
                 .sort_values('gini_importance', ascending=False)
        print("\nImportances (impureté):\n", df_imp.to_string(index=False))
    else:
        df_imp = None

    # Permutation Importance (sur test): plus coûteux mais moins biaisé
    try:
        X_test_trans = pre.transform(X_test)
        perm = permutation_importance(clf, X_test_trans, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        df_perm = pd.DataFrame({'feature': feat_names, 'perm_importance': perm.importances_mean}) \
                   .sort_values('perm_importance', ascending=False)
        print("\nImportances (permutation, test):\n", df_perm.to_string(index=False))
    except Exception as e:
        print("Permutation importance non calculée:", e)
        df_perm = None

    return df_imp, df_perm


def main(csv: str | None = None, df: pd.DataFrame | None = None):
    df0 = load_data(csv, df)

    # Définition des colonnes
    numeric_cols = [
        'age','height_cm','weight_kg','heart_rate','blood_pressure',
        'sleep_hours','nutrition_quality','activity_index'
    ]
    binary_cols = ['smokes','gender']

    splits = split_data(df0)

    models, best_params = fit_models(
        splits,
        feature_cols={'num': numeric_cols, 'bin': binary_cols}
    )

    print("\nBest params:", best_params)

    leaderboard = evaluate_models(models, splits)
    print("\nLeaderboard (trié par F1):\n", leaderboard.to_string(index=False))

    # Interprétation LR & RF
    _ = interpret_lr(models['lr'], feature_cols={'num': numeric_cols, 'bin': binary_cols})
    _ = interpret_rf(models['rf'], splits.X_test, splits.y_test, feature_cols={'num': numeric_cols, 'bin': binary_cols})

    return models, leaderboard


if __name__ == "__main__":
    models, leaderboard = main(csv="/Users/alexandrebredillot/Documents/GitHub/EXP/BICLASS - Fitness/fitness_dataset.csv")
