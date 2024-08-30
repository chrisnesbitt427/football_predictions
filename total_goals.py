"""
Football Prediction Script

Script to predict the outcome of football matches
"""

__date__ = "2024-07-30"
__author__ = ""



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error

import optuna
import lightgbm as lgb

# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# %% --------------------------------------------------------------------------
# Load Whole Dataset
# -----------------------------------------------------------------------------
total_df = pd.read_csv("data/no_first_5_match_data.csv")

# %% --------------------------------------------------------------------------
# Column to Predict
# -----------------------------------------------------------------------------
col_to_predict = "home_Gls"
drop = ['home_Gls','Away_Gls', 'Round', 'round_num']

# %% --------------------------------------------------------------------------
# Data Split
# -----------------------------------------------------------------------------
X = total_df.drop(columns=drop)
y = total_df[col_to_predict]

# %% --------------------------------------------------------------------------
# Train Test Split
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rng
)

# %% --------------------------------------------------------------------------
# Hyperparameter Tune
# -----------------------------------------------------------------------------
def objective(trial):
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'objective': 'regression',
        'max_depth': trial.suggest_int('max_depth', -1, 20),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 50)
    }
    
    model = lgb.LGBMRegressor(**param)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    
    return np.sqrt(-score.mean())

study = optuna.create_study(direction='minimize')
study.optimize(objective, timeout=300)

# %% --------------------------------------------------------------------------
# Hyperparameter Tune Overnight
# -----------------------------------------------------------------------------
def objective(trial):
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 40, 60, 80, 100, 150),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.01, 0.05, 0.1, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 200, 500, 1000, 2000),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'objective': 'regression',
        'max_depth': trial.suggest_int('max_depth', -1, 5, 10, 15, 20, 30),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 20, 30, 40, 50)
    }
    
    model = lgb.LGBMRegressor(**param)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    
    return np.sqrt(-score.mean())

study = optuna.create_study(direction='minimize')
study.optimize(objective, timeout=36000)

# %% --------------------------------------------------------------------------
# Initialise Model
# -----------------------------------------------------------------------------
best_model = lgb.LGBMRegressor(**study.best_params)
best_model.fit(X_train, y_train)

# %% --------------------------------------------------------------------------
# See Model Quality
# -----------------------------------------------------------------------------
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Test RMSE: ", rmse)


# %% --------------------------------------------------------------------------
# Review
# -----------------------------------------------------------------------------
display(lgb.plot_importance(best_model))
