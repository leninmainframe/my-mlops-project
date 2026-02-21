"""
scripts/run_pipeline_all.py
============================
Master orchestrator — runs ALL pipeline steps in sequence:

  Step 1 → Upload used_cars.csv to AzureML datastore
  Step 2 → Register Azure ML Environment  (pricing-env)
  Step 3 → Register pipeline Components  (preprocess / train_tune / register_model)
  Step 4 → Submit the end-to-end AzureML Pipeline job

Run locally (after az login):
    python scripts/run_pipeline_all.py

Run in CI/CD (GitHub Actions):
    python scripts/run_pipeline_all.py
    (env vars AZURE_CLIENT_ID, AZURE_CLIENT_SECRET etc. are injected by the workflow)

Optional environment variable overrides:
    DATA_PATH   – override the AzureML data URI  (default: uses registered asset)
    COMPUTE     – override compute cluster name   (default: from config.json)
    SKIP_UPLOAD – set to '1' to skip data upload  (useful if already uploaded)
"""

import os
import sys
import time
from pathlib import Path

# ── Add scripts/ to path so we can import local modules ───────────────────
sys.path.insert(0, str(Path(__file__).parent))

from auth_helper import load_config, get_ml_client

from azure.ai.ml import Input, load_component, load_environment
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Data, Model
from azure.ai.ml.exceptions import ValidationException
from azure.core.exceptions import HttpResponseError

ROOT = Path(__file__).resolve().parent.parent

# ── Component versions to register and use ───────────────────────────────────
COMPONENT_YAMLS = {
    "preprocess_component":     "components/preprocess_component.yml",
    "train_tune_component":     "components/train_tune_component.yml",
    "register_model_component": "components/register_model_component.yml",
}
COMPONENT_VERSION  = "3"
DATA_ASSET_NAME    = "used_cars_raw"
DATA_ASSET_VERSION = "1"
FALLBACK_DATA_PATH = "azureml://datastores/workspaceblobstore/paths/used_cars.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def banner(step: int, title: str) -> None:
    print(f"\n{'═' * 55}")
    print(f"  STEP {step}: {title}")
    print(f"{'═' * 55}")


def elapsed(start: float) -> str:
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Upload dataset
# ─────────────────────────────────────────────────────────────────────────────
def upload_dataset(ml_client) -> str:
    """
    Upload used_cars.csv and register it as a Data Asset.
    Returns the azureml:<name>:<version> URI to pass into the pipeline.
    """
    if os.environ.get("SKIP_UPLOAD") == "1":
        print("[data]  SKIP_UPLOAD=1 — skipping upload, resolving existing asset …")
    else:
        csv_path = ROOT / "used_cars.csv"
        if not csv_path.exists():
            print(f"[ERROR] Dataset not found: {csv_path}")
            sys.exit(1)

        print(f"[data]  Uploading: {csv_path}  ({csv_path.stat().st_size / 1024:.1f} KB)")
        data_asset = Data(
            path=str(csv_path),
            type=AssetTypes.URI_FILE,
            name=DATA_ASSET_NAME,
            description="Raw used-car pricing CSV. Uploaded by run_pipeline_all.py.",
        )
        registered = ml_client.data.create_or_update(data_asset)
        print(f"[data]  ✅ Registered  {registered.name}  (version {registered.version})")
        print(f"[data]     URI  → {registered.path}")

    # Resolve the registered asset URI
    try:
        asset = ml_client.data.get(DATA_ASSET_NAME, version=DATA_ASSET_VERSION)
        uri = f"azureml:{asset.name}:{asset.version}"
        print(f"[data]  Using data URI: {uri}")
        return uri
    except Exception as exc:
        print(f"[data]  ⚠️  Could not resolve asset: {exc}")
        print(f"[data]  Falling back to: {FALLBACK_DATA_PATH}")
        return FALLBACK_DATA_PATH


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Register environment
# ─────────────────────────────────────────────────────────────────────────────
def register_environment(ml_client) -> None:
    env_yaml = ROOT / "environment.yaml"
    if not env_yaml.exists():
        print(f"[ERROR] environment.yaml not found at {env_yaml}")
        sys.exit(1)

    print(f"[env]   Loading: {env_yaml}")
    env_def = load_environment(source=str(env_yaml))
    registered = ml_client.environments.create_or_update(env_def)
    print(f"[env]   ✅ Registered  '{registered.name}'  (version {registered.version})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Register components
# ─────────────────────────────────────────────────────────────────────────────
def register_components(ml_client) -> None:
    for name, rel_path in COMPONENT_YAMLS.items():
        yaml_path = ROOT / rel_path
        if not yaml_path.exists():
            print(f"  [ERROR] YAML not found: {yaml_path}")
            sys.exit(1)
        comp = load_component(source=str(yaml_path))
        r    = ml_client.components.create_or_update(comp)
        print(f"  ✅ {r.name}  (version {r.version})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Build and submit pipeline
# ─────────────────────────────────────────────────────────────────────────────
def submit_pipeline(ml_client, cfg: dict, data_uri: str) -> None:
    # Load registered components
    print("[pipeline] Loading registered components …")
    try:
        preprocess_comp = ml_client.components.get(
            "preprocess_component", version=COMPONENT_VERSION)
        train_comp      = ml_client.components.get(
            "train_tune_component", version=COMPONENT_VERSION)
        register_comp   = ml_client.components.get(
            "register_model_component", version=COMPONENT_VERSION)
    except HttpResponseError as exc:
        print(f"[ERROR] Could not load components: {exc}")
        sys.exit(1)

    print("  ✅ preprocess_component      loaded")
    print("  ✅ train_tune_component      loaded")
    print("  ✅ register_model_component  loaded")

    # Pipeline DAG
    @pipeline(
        name="used_car_pricing_pipeline",
        description="End-to-end MLOps: preprocess → train+tune (MLflow) → register model",
    )
    def pricing_pipeline(raw_data: Input):
        # A. Preprocess raw CSV → processed_data.csv
        preprocess_step = preprocess_comp(input_data=raw_data)

        # B. Train + Optuna HPO + MLflow nested run logging → model.joblib
        train_step = train_comp(
            processed_data=preprocess_step.outputs.output_data
        )

        # C. Register model in AML Model Registry
        register_comp(model_path=train_step.outputs.model_folder)  # noqa

        return {
            "output_data":  preprocess_step.outputs.output_data,
            "model_folder": train_step.outputs.model_folder,
        }

    # Build pipeline job
    compute_name = os.environ.get("COMPUTE", cfg.get("compute"))
    data_path    = os.environ.get("DATA_PATH", data_uri)   # env-var final override

    print(f"\n[pipeline] data_path : {data_path}")
    print(f"[pipeline] compute   : {compute_name or 'serverless'}")

    pipeline_job = pricing_pipeline(
        raw_data=Input(
            type=AssetTypes.URI_FILE,
            path=data_path,
            mode="ro_mount",
        )
    )

    if compute_name:
        pipeline_job.settings.default_compute = compute_name

    # Submit
    print("\n[pipeline] Submitting job to Azure ML …")
    try:
        submitted = ml_client.jobs.create_or_update(
            pipeline_job,
            experiment_name="used-car-pricing",
        )
    except (ValidationException, HttpResponseError) as exc:
        print(f"[ERROR] Submission failed: {exc}")
        sys.exit(1)

    studio_url = (
        f"https://ml.azure.com/runs/{submitted.name}"
        f"?wsid=/subscriptions/{cfg['subscription_id']}"
        f"/resourceGroups/{cfg['resource_group']}"
        f"/providers/Microsoft.MachineLearningServices/workspaces/{cfg['workspace_name']}"
    )

    print(f"\n{'═' * 55}")
    print(f"  ✅ PIPELINE SUBMITTED SUCCESSFULLY")
    print(f"{'═' * 55}")
    print(f"  Job name   : {submitted.name}")
    print(f"  Experiment : used-car-pricing")
    print(f"  Status     : {submitted.status}")
    print(f"  Studio URL : {studio_url}")
    print(f"{'═' * 55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    total_start = time.time()
    print("\n🚀  Azure ML MLOps Pipeline — Full Run")
    print(f"{'═' * 55}")

    # ── Connect once, reuse the client for all steps ──────────────────────
    cfg       = load_config()
    ml_client = get_ml_client(cfg)

    # ── Step 1: Upload dataset ────────────────────────────────────────────
    banner(1, "Upload Dataset")
    t = time.time()
    data_uri = upload_dataset(ml_client)
    print(f"[step 1] Done in {elapsed(t)}")

    # ── Step 2: Register environment ──────────────────────────────────────
    banner(2, "Register Azure ML Environment")
    t = time.time()
    register_environment(ml_client)
    print(f"[step 2] Done in {elapsed(t)}")

    # ── Step 3: Register components ───────────────────────────────────────
    banner(3, "Register Pipeline Components")
    t = time.time()
    register_components(ml_client)
    print(f"[step 3] Done in {elapsed(t)}")

    # ── Step 4: Submit pipeline ───────────────────────────────────────────
    banner(4, "Submit AzureML Pipeline Job")
    t = time.time()
    submit_pipeline(ml_client, cfg, data_uri)
    print(f"[step 4] Done in {elapsed(t)}")

    print(f"✅  Total time: {elapsed(total_start)}")


if __name__ == "__main__":
    main()
