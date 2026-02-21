#!/usr/bin/env python
"""
register_model.py
=================
Component 3 — Model Registration
----------------------------------
• Reads the model artifact produced by train_tune.py
• Registers it in the Azure ML Model Registry using the AzureML job context
  (inside an AzureML job, MLClient can be created credential-free via the
   managed identity that the compute node runs under)
• Logs registration details to MLflow

Args (injected by AzureML component YAML):
    --model_path   : URI_FOLDER – folder containing model.joblib + model_metadata.json
    --model_name   : str        – name to register in AML model registry
    --model_version: str        – optional version tag (default: auto)
"""

import argparse
import json
import os

import mlflow
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Register trained model in AML Registry")
parser.add_argument("--model_path",    type=str, required=True,
                    help="URI_FOLDER containing model.joblib")
parser.add_argument("--model_name",    type=str, default="used_cars_pricing_model",
                    help="Name to register in AML model registry")
parser.add_argument("--model_version", type=str, default=None,
                    help="Version tag (leave empty for auto-increment)")
# Workspace coordinates — injected by execute_pipeline.py as env vars,
# OR picked up automatically via AzureML's AZUREML_* env vars on the compute node
parser.add_argument("--subscription_id",  type=str,
                    default=os.environ.get("AZURE_SUBSCRIPTION_ID",
                            os.environ.get("AZ_SUBSCRIPTION_ID")))
parser.add_argument("--resource_group",   type=str,
                    default=os.environ.get("AZ_RESOURCE_GROUP"))
parser.add_argument("--workspace_name",   type=str,
                    default=os.environ.get("AZ_WORKSPACE"))
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1. Resolve model path (file or folder)
# ---------------------------------------------------------------------------
model_path = args.model_path
if os.path.isdir(model_path):
    joblib_files = [f for f in os.listdir(model_path) if f.endswith(".joblib")]
    print(f"[register] Contents of model folder: {os.listdir(model_path)}")
else:
    print(f"[register] Model path is a file: {model_path}")

# Load metadata if available
meta_file = os.path.join(model_path, "model_metadata.json") if os.path.isdir(model_path) else None
meta = {}
if meta_file and os.path.exists(meta_file):
    with open(meta_file) as fh:
        meta = json.load(fh)
    print(f"[register] Loaded metadata: {json.dumps(meta, indent=2)}")

# ---------------------------------------------------------------------------
# 2. Build MLClient (runs on managed identity inside AzureML compute)
# ---------------------------------------------------------------------------
print("[register] Attempting to authenticate with DefaultAzureCredential …")
try:
    # On AzureML compute: uses managed identity automatically
    # Locally: uses az login / env vars
    credential = DefaultAzureCredential()
except Exception as exc:
    print(f"[register] DefaultAzureCredential failed ({exc}), trying ManagedIdentityCredential …")
    credential = ManagedIdentityCredential()

if args.subscription_id and args.resource_group and args.workspace_name:
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )
    print(f"[register] Connected to workspace: {args.workspace_name}")

    # ------------------------------------------------------------------
    # 3. Register model in AML Model Registry
    # ------------------------------------------------------------------
    description = (
        f"Used-car pricing model. "
        f"MAE={meta.get('test_mae', 'N/A'):.2f}  "
        f"RMSE={meta.get('test_rmse', 'N/A'):.2f}  "
        f"R²={meta.get('test_r2', 'N/A'):.4f}"
        if meta else "Used-car pricing GradientBoosting model"
    )
    model_entity = Model(
        path=model_path,
        name=args.model_name,
        description=description,
        type=AssetTypes.CUSTOM_MODEL,
        tags={
            "framework":   "scikit-learn",
            "task":        "regression",
            "dataset":     "used_cars",
            "mae":         str(round(meta.get("test_mae",  0), 4)),
            "rmse":        str(round(meta.get("test_rmse", 0), 4)),
            "r2":          str(round(meta.get("test_r2",   0), 6)),
        },
    )
    registered = ml_client.models.create_or_update(model_entity)
    print(f"[register] ✅ Registered model: {registered.name} (version {registered.version})")

    # ------------------------------------------------------------------
    # 4. Log registration result to MLflow
    # ------------------------------------------------------------------
    mlflow.start_run()
    mlflow.log_param("registered_model_name",    registered.name)
    mlflow.log_param("registered_model_version", registered.version)
    mlflow.log_metric("test_mae",  meta.get("test_mae",  0))
    mlflow.log_metric("test_rmse", meta.get("test_rmse", 0))
    mlflow.log_metric("test_r2",   meta.get("test_r2",   0))
    mlflow.end_run()

    result = {
        "registered_model_name":    registered.name,
        "registered_model_version": registered.version,
        "description":              description,
    }
    print(json.dumps(result, indent=2))

else:
    # Fallback: model artifact is already registered by MLflow in train_tune.py
    print("[register] ⚠️  Workspace coordinates not available — "
          "skipping explicit AML registry registration.")
    print("[register] ℹ️  Model was already logged to MLflow by train_tune.py "
          "(registered_model_name=used_cars_pricing_model).")

print("[register] ✅ Done.")
