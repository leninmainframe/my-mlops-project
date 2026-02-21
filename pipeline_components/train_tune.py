#!/usr/bin/env python
"""
train_tune.py
=============
Component 2 — Model Training & Hyperparameter Tuning with MLflow
-----------------------------------------------------------------
• Loads processed CSV produced by preprocess.py
• Runs Optuna hyperparameter search (n_trials configurable)
• Each trial is logged as a nested MLflow run (params + metrics)
• Trains a final GradientBoostingRegressor with best params
• Logs the final model artifact to MLflow model registry
• Saves model.joblib to the AzureML output folder

Args (injected by AzureML component YAML):
    --processed_data   : URI_FOLDER – folder containing processed_data.csv
    --model_folder     : URI_FOLDER – destination for model.joblib & metadata
    --experiment_name  : str        – MLflow experiment name (default: pricing_experiment)
    --trials           : int        – number of Optuna trials (default: 20)
"""

import argparse
import json
import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Suppress verbose Optuna trial logs; keep it readable
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train & tune a GradientBoosting model")
parser.add_argument("--processed_data",  type=str, required=True,
                    help="URI_FOLDER containing processed_data.csv")
parser.add_argument("--model_folder",    type=str, required=True,
                    help="URI_FOLDER to save model artifact")
parser.add_argument("--experiment_name", type=str, default="pricing_experiment",
                    help="MLflow experiment name")
parser.add_argument("--trials",          type=int, default=20,
                    help="Number of Optuna hyperparameter search trials")
args = parser.parse_args()

os.makedirs(args.model_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load processed data
# ---------------------------------------------------------------------------
data_path = (
    os.path.join(args.processed_data, "processed_data.csv")
    if os.path.isdir(args.processed_data)
    else args.processed_data
)
print(f"[train] Loading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"[train] Dataset shape: {df.shape}")

# Feature selection — only use columns that exist
FEATURE_CANDIDATES = [
    "Kilometers_Driven", "Mileage", "Engine", "Power",
    "Seats", "power_per_cc", "Segment_enc",
]
features = [f for f in FEATURE_CANDIDATES if f in df.columns]
TARGET    = "log_price"

X = df[features].fillna(0)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"[train] Features used: {features}")
print(f"[train] Train size: {len(X_train)}, Test size: {len(X_test)}")

# ---------------------------------------------------------------------------
# 2. MLflow experiment setup
# ---------------------------------------------------------------------------
mlflow.set_experiment(args.experiment_name)


def _metrics(y_true_log, y_pred_log):
    """Return MAE, RMSE, R² on the original price scale."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))   # sklearn ≥1.4 removed squared kwarg
    r2   = r2_score(y_true_log, y_pred_log)             # R² on log scale (model optimisation)
    return float(mae), float(rmse), float(r2)


# ---------------------------------------------------------------------------
# 3. Optuna hyperparameter search with nested MLflow runs
# ---------------------------------------------------------------------------
print(f"\n[train] Starting Optuna search ({args.trials} trials) …")

with mlflow.start_run(run_name="optuna_search") as parent_run:
    mlflow.log_param("n_trials",        args.trials)
    mlflow.log_param("features",        json.dumps(features))
    mlflow.log_param("train_size",      len(X_train))
    mlflow.log_param("test_size",       len(X_test))

    def objective(trial):
        params = {
            "n_estimators":  trial.suggest_int("n_estimators",  50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":     trial.suggest_int("max_depth",     2, 8),
            "subsample":     trial.suggest_float("subsample",   0.5, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            model = GradientBoostingRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            mae, rmse, r2 = _metrics(y_test.values, model.predict(X_test))
            mlflow.log_params(params)
            mlflow.log_metric("mae",  mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2",   r2)
        return mae   # minimise MAE

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)
    best_params = study.best_params
    print(f"[train] Best params: {best_params}")
    print(f"[train] Best trial MAE: {study.best_value:.4f}")

    # Log best params to parent run for easy reference
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_trial_mae", study.best_value)

# ---------------------------------------------------------------------------
# 4. Train final model with best hyperparameters
# ---------------------------------------------------------------------------
print("\n[train] Training final model with best hyperparameters …")
final_model = GradientBoostingRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)

mae_test, rmse_test, r2_test = _metrics(y_test.values, final_model.predict(X_test))
print(f"[train] Final model — MAE: {mae_test:.2f}  RMSE: {rmse_test:.2f}  R²: {r2_test:.4f}")

# ---------------------------------------------------------------------------
# 5. Log final model to MLflow (model registry + artifact)
# ---------------------------------------------------------------------------
model_file = os.path.join(args.model_folder, "model.joblib")
joblib.dump(final_model, model_file)

# Save feature list for downstream inference
meta = {
    "features":    features,
    "target":      TARGET,
    "best_params": best_params,
    "test_mae":    mae_test,
    "test_rmse":   rmse_test,
    "test_r2":     r2_test,
}
meta_file = os.path.join(args.model_folder, "model_metadata.json")
with open(meta_file, "w") as fh:
    json.dump(meta, fh, indent=2)

with mlflow.start_run(run_name="final_model") as final_run:
    mlflow.log_params(best_params)
    mlflow.log_metric("mae",  mae_test)
    mlflow.log_metric("rmse", rmse_test)
    mlflow.log_metric("r2",   r2_test)
    # Log model to MLflow registry
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        registered_model_name="used_cars_pricing_model",
        input_example=X_train.head(5),
    )
    mlflow.log_artifact(model_file,  artifact_path="joblib")
    mlflow.log_artifact(meta_file,   artifact_path="metadata")
    print(f"[train] MLflow run_id: {final_run.info.run_id}")

print(json.dumps({
    "model_file":  model_file,
    "mae":         mae_test,
    "rmse":        rmse_test,
    "r2":          r2_test,
    "best_params": best_params,
}, indent=2))
print("[train] ✅ Done.")
