"""
scripts/register_components.py
================================
Registers (or updates) all Azure ML pipeline components defined under
the components/ directory.

Local usage (after `az login` or SP login):
    python scripts/register_components.py

CI/CD usage (GitHub Actions):
    Env vars are injected by the workflow; auth_helper picks ClientSecretCredential.
"""

import os
import sys
from pathlib import Path

# Shared auth/config helper
sys.path.insert(0, str(Path(__file__).parent))
from auth_helper import load_config, get_ml_client

from azure.ai.ml import load_component
from azure.ai.ml.exceptions import ValidationException
from azure.core.exceptions import HttpResponseError

ROOT = Path(__file__).resolve().parent.parent  # repo root

# Component YAML files relative to repo root (in registration order)
COMPONENT_YAMLS = [
    "components/preprocess_component.yml",
    "components/train_tune_component.yml",
    "components/register_model_component.yml",
]


def main() -> None:
    cfg = load_config()
    ml_client = get_ml_client(cfg)

    # ------------------------------------------------------------------
    # Register each component
    # ------------------------------------------------------------------
    print("\n[components] Starting component registration …\n")
    all_ok = True
    for rel_path in COMPONENT_YAMLS:
        yaml_path = ROOT / rel_path
        if not yaml_path.exists():
            print(f"  [WARN] Component YAML not found – skipping: {yaml_path}")
            all_ok = False
            continue

        try:
            component_def = load_component(source=str(yaml_path))
            registered    = ml_client.components.create_or_update(component_def)
            print(f"  ✅ Registered '{registered.name}' (version: {registered.version})")
        except (ValidationException, HttpResponseError) as exc:
            print(f"  [ERROR] Failed to register {rel_path}: {exc}")
            all_ok = False

    if not all_ok:
        print("\n[ERROR] One or more components failed to register.")
        sys.exit(1)

    print("\n[components] ✅ All components registered successfully.")


if __name__ == "__main__":
    main()
