#!/usr/bin/env python
"""
train_tune.py
- Loads processed CSV
- Runs Optuna tuning loop and trains final model
- Logs hyperparams and metrics using MLflow
- Saves model artifact to output folder
"""
import argparse, os, json
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib, mlflow, mlflow.sklearn, optuna

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str, required=True, help="Processed CSV (file)")
parser.add_argument("--output_model", type=str, required=True, help="Output model folder")
parser.add_argument("--experiment_name", type=str, default="pricing_experiment")
parser.add_argument("--trials", type=int, default=20)
args = parser.parse_args()

os.makedirs(args.output_model, exist_ok=True)

# Read processed data
df = pd.read_csv(args.input_data)

# Define features/target
features = ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats", "power_per_cc", "Segment_enc"]
features = [f for f in features if f in df.columns]
X = df[features].fillna(0)
y = df["log_price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow experiment
mlflow.set_experiment(args.experiment_name)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }
    with mlflow.start_run(nested=True):
        model = GradientBoostingRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        # compute metrics on original scale
        mae = mean_absolute_error(np.expm1(y_test), np.expm1(preds))
        rmse = mean_squared_error(np.expm1(y_test), np.expm1(preds), squared=False)
        r2 = r2_score(y_test, preds)
        mlflow.log_params(params)
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("r2", float(r2))
    return mae

# Run Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=args.trials)

best_params = study.best_params

# Train final model
final_model = GradientBoostingRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)
preds_test = final_model.predict(X_test)
mae_test = mean_absolute_error(np.expm1(y_test), np.expm1(preds_test))
rmse_test = mean_squared_error(np.expm1(y_test), np.expm1(preds_test), squared=False)
r2_test = r2_score(y_test, preds_test)

# Save final model and log to mlflow
model_file = os.path.join(args.output_model, "model.joblib")
joblib.dump(final_model, model_file)

with mlflow.start_run(run_name="final_model"):
    mlflow.log_params(best_params)
    mlflow.log_metric("mae", float(mae_test))
    mlflow.log_metric("rmse", float(rmse_test))
    mlflow.log_metric("r2", float(r2_test))
    mlflow.sklearn.log_model(final_model, "model")
    mlflow.log_artifact(model_file)

print(json.dumps({"model_path": model_file, "mae": mae_test, "rmse": rmse_test, "r2": r2_test}))
