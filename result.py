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
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import optuna
import lightgbm as lgb

import datetime as dt

# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)


# %% --------------------------------------------------------------------------
# Run Name
# -----------------------------------------------------------------------------
run_name = '1hour'
how_long = 3600

# %% --------------------------------------------------------------------------
# %% --------------------------------------------------------------------------
# Load Whole Dataset
# -----------------------------------------------------------------------------
total_df = pd.read_csv("data/to_win.csv")

# %% --------------------------------------------------------------------------
# Column to Predict
# -----------------------------------------------------------------------------
col_to_predict = "f0_"
drop = ['f0_','home_Gls', 'Away_Gls', 'Round', 'round_num']

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
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
        'objective': 'binary',  
        'max_depth': trial.suggest_int('max_depth', -1, 30),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100)
    }
    
    model = lgb.LGBMClassifier(**param)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=rng)
    score = cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc')
    
    return score.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, show_progress_bar=True, timeout=how_long)


# %% --------------------------------------------------------------------------
# Initialise Model
# -----------------------------------------------------------------------------
best_model = lgb.LGBMClassifier(**study.best_params)
best_model.fit(X_train, y_train)

# %% --------------------------------------------------------------------------
# Calcuated Model Statistics
# -----------------------------------------------------------------------------
y_pred_prob = best_model.predict_proba(X_test)
y_pred_prob_pos = y_pred_prob[:, 1]

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob_pos)

confidences = np.max(y_pred_prob, axis=1)  # Maximum probability for the predicted class
mean_confidence = np.mean(confidences)

calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid')
calibrated_model.fit(X_train, y_train)

y_calibrated_prob = calibrated_model.predict_proba(X_test)
y_calibrated_prob_pos = y_calibrated_prob[:, 1]

# %% --------------------------------------------------------------------------
# Output Folder
# -----------------------------------------------------------------------------
output_folder = f"output/{run_name}"
os.makedirs(output_folder, exist_ok=True)

# %% --------------------------------------------------------------------------
# Save Data
# -----------------------------------------------------------------------------
# Model
model_path = f"{output_folder}/model.pkl"

with open(model_path, 'wb') as file:
    pickle.dump(best_model,file)

# Feature Importance
importance = lgb.plot_importance(best_model)
importance.figure.savefig(f"{output_folder}/feature_importance.png")

# Metrics
metrics_df = pd.DataFrame({
    'Accuracy': [accuracy],
    'ROC AUC': [roc_auc],
    'Mean Confidence': [mean_confidence]
})
metrics_df.to_csv(f"{output_folder}/metrics.csv")

# Calibrated Confidence
calibrated_confidences = np.max(y_calibrated_prob, axis=1)
plt.hist(calibrated_confidences, bins=20, edgecolor='k')
plt.xlabel('Calibrated Confidence Score')
plt.ylabel('Frequency')
plt.title('Distribution of Calibrated Confidence Scores')
plt.show()
plt.savefig(f"{output_folder}/calibrated_confidence_scores.png")

# Confidence
confidence_plot = plt.hist(confidences, bins=20, edgecolor='k')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Distribution of Confidence Scores')
plt.show()
plt.savefig(f"{output_folder}/confidence_scores.png")

print("Accuracy: ", accuracy)
print("ROC-AUC Score: ", roc_auc)
print("Mean confidence of the model: ", mean_confidence)

# %%
