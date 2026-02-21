"""
scripts/run_pipeline_all.py
Master orchestrator — runs ALL pipeline steps in sequence
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from auth_helper import load_config, get_ml_client

from azure.ai.ml import Input, load_component, load_environment
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Data
from azure.ai.ml.exceptions import ValidationException
from azure.core.exceptions import HttpResponseError

ROOT = Path(__file__).resolve().parent.parent

COMPONENT_YAMLS = {
    "preprocess_component": "components/preprocess_component.yml",
    "train_tune_component": "components/train_tune_component.yml",
    "register_model_component": "components/register_model_component.yml",
}

COMPONENT_VERSION = "3"
DATA_ASSET_NAME = "used_cars_raw"
DATA_ASSET_VERSION = "1"
FALLBACK_DATA_PATH = "azureml://datastores/workspaceblobstore/paths/used_cars.csv"


def banner(step: int, title: str):
    print(f"\n{'═'*55}")
    print(f"  STEP {step}: {title}")
    print(f"{'═'*55}")


def elapsed(start):
    s = int(time.time() - start)
    return f"{s//60}m {s%60}s"


# ─────────────────────────────────────────────────────────────
# STEP 1 — Upload dataset
# ─────────────────────────────────────────────────────────────
def upload_dataset(ml_client):
    if os.environ.get("SKIP_UPLOAD") == "1":
        print("[data] SKIP_UPLOAD=1 — using existing asset")
    else:
        csv_path = ROOT / "used_cars.csv"
        if not csv_path.exists():
            sys.exit(f"[ERROR] Dataset not found: {csv_path}")

        print(f"[data] Uploading: {csv_path}")
        data_asset = Data(
            path=str(csv_path),
            type=AssetTypes.URI_FILE,
            name=DATA_ASSET_NAME,
        )
        registered = ml_client.data.create_or_update(data_asset)
        print(f"[data] Registered {registered.name}:{registered.version}")

    try:
        asset = ml_client.data.get(DATA_ASSET_NAME, version=DATA_ASSET_VERSION)
        uri = f"azureml:{asset.name}:{asset.version}"
        print(f"[data] Using data URI: {uri}")
        return uri
    except Exception:
        print("[data] Using fallback path")
        return FALLBACK_DATA_PATH


# ─────────────────────────────────────────────────────────────
# STEP 2 — Register environment
# ─────────────────────────────────────────────────────────────
def register_environment(ml_client):
    env_yaml = ROOT / "environment.yaml"
    env_def = load_environment(source=str(env_yaml))
    ml_client.environments.create_or_update(env_def)
    print("[env] Environment registered")


# ─────────────────────────────────────────────────────────────
# STEP 3 — Register components
# ─────────────────────────────────────────────────────────────
def register_components(ml_client):
    for name, path in COMPONENT_YAMLS.items():
        comp = load_component(source=str(ROOT / path))
        ml_client.components.create_or_update(comp)
        print(f"[comp] {name} registered")


# ─────────────────────────────────────────────────────────────
# STEP 4 — Submit pipeline
# ─────────────────────────────────────────────────────────────
def submit_pipeline(ml_client, cfg, data_uri):

    preprocess_comp = ml_client.components.get("preprocess_component", version=COMPONENT_VERSION)
    train_comp = ml_client.components.get("train_tune_component", version=COMPONENT_VERSION)
    register_comp = ml_client.components.get("register_model_component", version=COMPONENT_VERSION)

    @pipeline()
    def pricing_pipeline(raw_data: Input):
        p = preprocess_comp(input_data=raw_data)
        t = train_comp(processed_data=p.outputs.output_data)
        register_comp(model_path=t.outputs.model_folder)
        return {}

    compute = os.environ.get("COMPUTE") or cfg.get("compute")

    # ⭐⭐⭐ FIXED LINE ⭐⭐⭐
    data_path = os.environ.get("DATA_PATH") or data_uri

    print("[pipeline] data_path:", data_path)
    print("[pipeline] compute:", compute)

    job = pricing_pipeline(
        raw_data=Input(type=AssetTypes.URI_FILE, path=data_path)
    )

    if compute:
        job.settings.default_compute = compute

    submitted = ml_client.jobs.create_or_update(job, experiment_name="used-car-pricing")

    print("\n✅ PIPELINE SUBMITTED")
    print("Job name:", submitted.name)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    cfg = load_config()
    ml_client = get_ml_client(cfg)

    banner(1, "Upload Dataset")
    data_uri = upload_dataset(ml_client)

    banner(2, "Register Environment")
    register_environment(ml_client)

    banner(3, "Register Components")
    register_components(ml_client)

    banner(4, "Submit Pipeline")
    submit_pipeline(ml_client, cfg, data_uri)


if __name__ == "__main__":
    main()
