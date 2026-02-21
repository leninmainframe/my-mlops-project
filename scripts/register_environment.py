"""
scripts/register_environment.py
================================
Registers (or updates) the Azure ML environment defined in environment.yaml.

Local usage (after `az login` or SP login):
    python scripts/register_environment.py

CI/CD usage (GitHub Actions):
    Env vars AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, etc. are injected by the
    workflow and auth_helper.py will automatically use ClientSecretCredential.
"""

import os
import sys
from pathlib import Path

# Shared auth/config helper
sys.path.insert(0, str(Path(__file__).parent))
from auth_helper import load_config, get_ml_client

from azure.ai.ml import load_environment
from azure.ai.ml.exceptions import ValidationException
from azure.core.exceptions import HttpResponseError

ROOT = Path(__file__).resolve().parent.parent  # repo root


def main() -> None:
    cfg = load_config()
    ml_client = get_ml_client(cfg)

    # ------------------------------------------------------------------
    # Load and register environment
    # ------------------------------------------------------------------
    env_yaml = ROOT / "environment.yaml"
    if not env_yaml.exists():
        print(f"[ERROR] environment.yaml not found at {env_yaml}")
        sys.exit(1)

    print(f"\n[env]  Loading environment definition from {env_yaml} …")
    try:
        env_def = load_environment(source=str(env_yaml))
        registered_env = ml_client.environments.create_or_update(env_def)
        print(f"[env]  ✅ Registered environment '{registered_env.name}' "
              f"(version: {registered_env.version})")
    except (ValidationException, HttpResponseError) as exc:
        print(f"[ERROR] Failed to register environment: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
