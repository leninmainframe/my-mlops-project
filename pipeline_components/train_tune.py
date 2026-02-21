#!/usr/bin/env python
"""
train_tune.py
=============
Component 2 — Model Training & Hyperparameter Tuning with MLflow
-----------------------------------------------------------------
• Loads processed CSV produced by preprocess.py
• Runs Optuna hyperparameter search (n_trials configurable)
• Each Optuna trial is logged as a NESTED MLflow run (params + metrics)
• Trains a final GradientBoostingRegressor with best params
• Logs the final model artifact to MLflow (WITHOUT registered_model_name —
  AzureML's model registry is handled separately by register_model.py)
• Saves model.joblib + model_metadata.json to the AzureML output folder

IMPORTANT — AzureML MLflow notes:
  - AzureML auto-starts an outer MLflow run for each job step.
  - Do NOT call mlflow.set_experiment() — AzureML sets it automatically.
  - Starting a NEW top-level run (mlflow.start_run() with no parent) is OK
    only when wrapping Optuna — use nested=True for trial runs.
  - Do NOT pass registered_model_name to mlflow.sklearn.log_model() —
    the azureml:// URI does not support the MLflow model registry API.
    Model registration is handled by register_model.py using the AML SDK.

Args (injected by AzureML component YAML):
    --processed_data   : URI_FOLDER – folder containing processed_data.csv
    --model_folder     : URI_FOLDER – destination for model.joblib & metadata
    --experiment_name  : str        – used only as a run tag (not set_experiment)
    --trials           : int        – number of Optuna trials (default: 20)
"""

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# MLflow — available inside AzureML jobs via azureml-mlflow bridge
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[train] WARNING: mlflow not installed – skipping MLflow logging")

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train & tune a GradientBoosting model")
parser.add_argument("--processed_data",  type=str, required=True)
parser.add_argument("--model_folder",    type=str, required=True)
parser.add_argument("--experiment_name", type=str, default="pricing_experiment")
parser.add_argument("--trials",          type=int, default=20)
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

FEATURE_CANDIDATES = [
    "Kilometers_Driven", "Mileage", "Engine", "Power",
    "Seats", "power_per_cc", "Segment_enc",
]
features = [f for f in FEATURE_CANDIDATES if f in df.columns]
TARGET   = "log_price"

X = df[features].fillna(0)
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"[train] Features: {features}")
print(f"[train] Train: {len(X_train)}  Test: {len(X_test)}")


def _metrics(y_true_log, y_pred_log):
    """MAE, RMSE, R² on the original price scale."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true_log, y_pred_log)
    return float(mae), float(rmse), float(r2)


# ---------------------------------------------------------------------------
# 2. Optuna HPO with MLflow nested run logging
#    Each trial → nested MLflow child run (parent is AzureML's auto-run)
# ---------------------------------------------------------------------------
print(f"\n[train] Starting Optuna search ({args.trials} trials) …")

# Tag the AzureML-managed run with experiment metadata
if MLFLOW_AVAILABLE:
    try:
        mlflow.set_tag("experiment_name", args.experiment_name)
        mlflow.log_param("n_trials",  args.trials)
        mlflow.log_param("features",  json.dumps(features))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size",  len(X_test))
    except Exception as exc:
        print(f"[train] WARNING: MLflow tag/param logging failed: {exc}")


def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators",      50, 300),
        "learning_rate":     trial.suggest_float("learning_rate",   0.01, 0.3, log=True),
        "max_depth":         trial.suggest_int("max_depth",         2, 8),
        "subsample":         trial.suggest_float("subsample",       0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    }
    model = GradientBoostingRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    mae, rmse, r2 = _metrics(y_test.values, model.predict(X_test))

    if MLFLOW_AVAILABLE:
        try:
            # nested=True → child of AzureML's managed parent run
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mae",  mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2",   r2)
        except Exception as exc:
            print(f"[train] WARNING: MLflow nested run failed (trial {trial.number}): {exc}")

    return mae


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=args.trials)
best_params = study.best_params
print(f"[train] Best params : {best_params}")
print(f"[train] Best MAE    : {study.best_value:.4f}")

# ---------------------------------------------------------------------------
# 3. Train final model with best hyperparameters
# ---------------------------------------------------------------------------
print("\n[train] Training final model …")
final_model = GradientBoostingRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)
mae_test, rmse_test, r2_test = _metrics(y_test.values, final_model.predict(X_test))
print(f"[train] Final — MAE: {mae_test:.2f}  RMSE: {rmse_test:.2f}  R²: {r2_test:.4f}")

# ---------------------------------------------------------------------------
# 4. Save model + metadata to output folder
# ---------------------------------------------------------------------------
model_file = os.path.join(args.model_folder, "model.joblib")
joblib.dump(final_model, model_file)

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

# ---------------------------------------------------------------------------
# 5. Log final model metrics + artifact to MLflow
#    NOTE: Do NOT use registered_model_name — AzureML URI does not support
#          the MLflow model registry API. Model registration is done by
#          register_model.py using the AML SDK instead.
# ---------------------------------------------------------------------------
if MLFLOW_AVAILABLE:
    try:
        # Log final metrics into the active AzureML run
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("final_mae",  mae_test)
        mlflow.log_metric("final_rmse", rmse_test)
        mlflow.log_metric("final_r2",   r2_test)

        # Log model as a scikit-learn artifact (NO registered_model_name)
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            # ⚠️  registered_model_name intentionally omitted —
            #     the azureml:// tracking URI doesn't support the MLflow
            #     model registry API. Use register_model.py + AML SDK instead.
        )
        mlflow.log_artifact(model_file,  artifact_path="joblib")
        mlflow.log_artifact(meta_file,   artifact_path="metadata")
        print("[train] MLflow metrics + model artifact logged to AzureML run.")
    except Exception as exc:
        print(f"[train] WARNING: MLflow final logging failed (non-fatal): {exc}")

print(json.dumps({
    "model_file":  model_file,
    "mae":         mae_test,
    "rmse":        rmse_test,
    "r2":          r2_test,
    "best_params": best_params,
}, indent=2))
print("[train] ✅ Done.")
