"""
scripts/execute_pipeline.py
============================
End-to-end script: loads registered components, builds the pipeline DAG,
and submits a job to Azure ML.

Pre-requisites:
  - python scripts/register_environment.py   (registers pricing-env)
  - python scripts/register_components.py    (registers all 3 components)

Local usage:
    python scripts/execute_pipeline.py

CI/CD usage (GitHub Actions):
    Env-vars injected by the workflow; auth_helper uses ClientSecretCredential.

Optional env-var overrides:
    DATA_PATH   – AzureML URI to raw CSV  (default: workspaceblobstore path)
    COMPUTE     – compute cluster name     (default: from config.json["compute"])
"""

import os
import sys
from pathlib import Path

# Shared auth/config helper
sys.path.insert(0, str(Path(__file__).parent))
from auth_helper import load_config, get_ml_client

from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.exceptions import ValidationException
from azure.core.exceptions import HttpResponseError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Registered Data Asset name (created by upload_data.py)
DATA_ASSET_NAME    = "used_cars_raw"
DATA_ASSET_VERSION = "1"   # bump when you upload a new version

# Fallback raw blob URI (only if the asset hasn't been registered yet)
FALLBACK_DATA_PATH = (
    "azureml://datastores/workspaceblobstore/paths/used_cars.csv"
)

# Component versions to load from the workspace registry
COMPONENT_VERSIONS = {
    "preprocess_component":      "3",
    "train_tune_component":      "3",
    "register_model_component":  "3",
}


# ---------------------------------------------------------------------------
# Resolve data path from registered Data Asset
# ---------------------------------------------------------------------------
def resolve_data_path(ml_client, env_override: str = "") -> str:
    """
    Priority:
      1. DATA_PATH env var  (explicit override)
      2. Registered Data Asset  azureml:used_cars_raw:<version>
      3. FALLBACK_DATA_PATH  (hardcoded blob path)
    """
    if env_override:
        print(f"[data]  Using DATA_PATH override: {env_override}")
        return env_override

    try:
        asset = ml_client.data.get(DATA_ASSET_NAME, version=DATA_ASSET_VERSION)
        uri = f"azureml:{asset.name}:{asset.version}"
        print(f"[data]  Resolved data asset  → {uri}")
        print(f"[data]  Blob path            → {asset.path}")
        return uri
    except Exception as exc:
        print(f"[data]  ⚠️  Could not resolve data asset '{DATA_ASSET_NAME}': {exc}")
        print(f"[data]  Run  python scripts/upload_data.py  to upload the dataset first.")
        print(f"[data]  Falling back to: {FALLBACK_DATA_PATH}")
        return FALLBACK_DATA_PATH


# ---------------------------------------------------------------------------
# Pipeline DSL definition
# ---------------------------------------------------------------------------
def build_pipeline(preprocess_comp, train_comp, register_comp, raw_data_path: str):
    """Construct the AzureML pipeline using the @pipeline DSL decorator."""

    @pipeline(
        name="used_car_pricing_pipeline",
        description="End-to-end MLOps: preprocess → train+tune (MLflow) → register model",
    )
    def pricing_pipeline(raw_data: Input):
        # Step 1: Preprocess raw data
        preprocess_step = preprocess_comp(input_data=raw_data)

        # Step 2: Train & tune model (Optuna + MLflow logging)
        train_step = train_comp(
            processed_data=preprocess_step.outputs.output_data
        )

        # Step 3: Register model in AML Model Registry
        register_step = register_comp(  # noqa: F841
            model_path=train_step.outputs.model_folder
        )

        return {
            "output_data":  preprocess_step.outputs.output_data,
            "model_folder": train_step.outputs.model_folder,
        }

    return pricing_pipeline(
        raw_data=Input(
            type=AssetTypes.URI_FILE,
            path=raw_data_path,
            mode="ro_mount",   # read-only mount — works with both blob URIs and azureml: URIs
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    ml_client = get_ml_client(cfg)

    # ------------------------------------------------------------------
    # Load registered components from workspace
    # ------------------------------------------------------------------
    print("\n[pipeline] Loading registered components …")
    try:
        preprocess_comp = ml_client.components.get(
            "preprocess_component",
            version=COMPONENT_VERSIONS["preprocess_component"],
        )
        train_comp = ml_client.components.get(
            "train_tune_component",
            version=COMPONENT_VERSIONS["train_tune_component"],
        )
        register_comp = ml_client.components.get(
            "register_model_component",
            version=COMPONENT_VERSIONS["register_model_component"],
        )
    except HttpResponseError as exc:
        print(f"[ERROR] Could not retrieve component from workspace: {exc}")
        print("  Make sure you've run `python scripts/register_components.py` first.")
        sys.exit(1)

    print("  ✅ preprocess_component      loaded")
    print("  ✅ train_tune_component      loaded")
    print("  ✅ register_model_component  loaded")

    # ------------------------------------------------------------------
    # Build pipeline job
    # ------------------------------------------------------------------
    # Resolve data path: registered asset URI > DATA_PATH env var > fallback blob path
    data_path    = resolve_data_path(ml_client, env_override=os.environ.get("DATA_PATH", ""))
    compute_name = os.environ.get("COMPUTE", cfg.get("compute"))

    print(f"\n[pipeline] Building pipeline …")
    print(f"  data_path : {data_path}")
    print(f"  compute   : {compute_name or 'serverless (default)'}")

    try:
        pipeline_job = build_pipeline(
            preprocess_comp, train_comp, register_comp, data_path
        )
    except (ValidationException, Exception) as exc:
        print(f"[ERROR] Pipeline build failed: {exc}")
        sys.exit(1)

    # Set compute target
    if compute_name:
        pipeline_job.settings.default_compute = compute_name

    # ------------------------------------------------------------------
    # Submit pipeline job
    # ------------------------------------------------------------------
    print("\n[pipeline] Submitting pipeline job …")
    try:
        submitted_job = ml_client.jobs.create_or_update(
            pipeline_job,
            experiment_name="used-car-pricing",
        )
    except (ValidationException, HttpResponseError) as exc:
        print(f"[ERROR] Pipeline submission failed: {exc}")
        sys.exit(1)

    studio_url = (
        f"https://ml.azure.com/runs/{submitted_job.name}"
        f"?wsid=/subscriptions/{cfg['subscription_id']}"
        f"/resourceGroups/{cfg['resource_group']}"
        f"/providers/Microsoft.MachineLearningServices/workspaces/{cfg['workspace_name']}"
    )

    print(f"\n[pipeline] ✅ Pipeline submitted successfully!")
    print(f"  Job name   : {submitted_job.name}")
    print(f"  Experiment : used-car-pricing")
    print(f"  Status     : {submitted_job.status}")
    print(f"  Studio URL : {studio_url}")


if __name__ == "__main__":
    main()
