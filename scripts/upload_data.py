"""
scripts/upload_data.py
=======================
Uploads used_cars.csv to the AzureML workspace's default datastore
and registers it as a versioned Data Asset.

Run this ONCE before submitting the pipeline (or whenever the dataset changes).

Local usage:
    python scripts/upload_data.py

CI/CD usage:
    Called automatically by the GitHub Actions workflow (setup job).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from auth_helper import load_config, get_ml_client

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cfg = load_config()
    ml_client = get_ml_client(cfg)

    # ------------------------------------------------------------------
    # Locate the local CSV file
    # ------------------------------------------------------------------
    csv_path = ROOT / "used_cars.csv"
    if not csv_path.exists():
        print(f"[ERROR] Dataset not found: {csv_path}")
        sys.exit(1)

    print(f"[data]  Uploading: {csv_path}  ({csv_path.stat().st_size / 1024:.1f} KB)")

    # ------------------------------------------------------------------
    # Register as a URI_FILE Data Asset in AzureML
    # create_or_update auto-increments the version each call
    # ------------------------------------------------------------------
    data_asset = Data(
        path=str(csv_path),
        type=AssetTypes.URI_FILE,
        name="used_cars_raw",
        description=(
            "Raw used-car pricing dataset. "
            "Contains Kilometers_Driven, Mileage, Engine, Power, Seats, Segment, price."
        ),
    )

    registered = ml_client.data.create_or_update(data_asset)

    print(f"[data]  ✅ Registered data asset!")
    print(f"         Name    : {registered.name}")
    print(f"         Version : {registered.version}")
    print(f"         URI     : {registered.path}")

    # ------------------------------------------------------------------
    # Print the URI to use in execute_pipeline.py
    # ------------------------------------------------------------------
    azureml_uri = f"azureml:{registered.name}:{registered.version}"
    print(f"\n[data]  Use this in execute_pipeline.py / DATA_PATH env var:")
    print(f"         {azureml_uri}")


if __name__ == "__main__":
    main()
