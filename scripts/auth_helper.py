"""
scripts/auth_helper.py
-----------------------
Shared credential + config helper used by all three pipeline scripts.

Auth priority:
  1. Service Principal (AZURE_CLIENT_ID + AZURE_CLIENT_SECRET + AZURE_TENANT_ID)
     → used in GitHub Actions CI/CD
  2. AzureCliCredential(tenant_id=...)
     → used locally after `az login --service-principal ...` or `az login`
  3. DefaultAzureCredential
     → fallback for Managed Identity / VS Code / etc.

Config priority:  env vars  >  config.json  (local dev fallback)
"""

import json
import os
import sys
from pathlib import Path

from azure.identity import (
    AzureCliCredential,
    ClientSecretCredential,
    DefaultAzureCredential,
)
from azure.ai.ml import MLClient

ROOT = Path(__file__).resolve().parent.parent  # repo root


# ──────────────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    """
    Merge config.json with environment variable overrides and return a dict
    with keys: subscription_id, resource_group, workspace_name, tenant_id,
               client_id, client_secret, compute.
    """
    cfg: dict = {}

    config_path = ROOT / "config.json"
    if config_path.exists():
        with open(config_path) as fh:
            cfg = json.load(fh)
        print(f"[config] Loaded base config from {config_path}")

    # Workspace coordinates
    cfg["subscription_id"] = os.environ.get("AZURE_SUBSCRIPTION_ID", cfg.get("subscription_id"))
    cfg["resource_group"]  = os.environ.get("AZ_RESOURCE_GROUP",     cfg.get("resource_group"))
    cfg["workspace_name"]  = os.environ.get("AZ_WORKSPACE",          cfg.get("workspace_name"))
    cfg["compute"]         = os.environ.get("COMPUTE",               cfg.get("compute"))

    # Auth coordinates
    cfg["tenant_id"]       = os.environ.get("AZURE_TENANT_ID",       cfg.get("tenant_id"))
    cfg["client_id"]       = os.environ.get("AZURE_CLIENT_ID",       cfg.get("client_id"))
    cfg["client_secret"]   = os.environ.get("AZURE_CLIENT_SECRET",   cfg.get("client_secret"))

    required = ["subscription_id", "resource_group", "workspace_name"]
    missing  = [k for k in required if not cfg.get(k)]
    if missing:
        print(f"[ERROR] Missing required config keys: {missing}")
        print("  Set them in config.json or via env vars: "
              "AZURE_SUBSCRIPTION_ID, AZ_RESOURCE_GROUP, AZ_WORKSPACE")
        sys.exit(1)

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
def get_credential(cfg: dict):
    """
    Return the best Azure credential available given the current environment.

    Priority:
      1. ClientSecretCredential  – when AZURE_CLIENT_SECRET is present (CI/CD)
      2. AzureCliCredential      – when tenant_id is set (local SP/user login)
      3. DefaultAzureCredential  – generic fallback
    """
    tenant_id     = cfg.get("tenant_id")
    client_id     = cfg.get("client_id")
    client_secret = cfg.get("client_secret")

    if client_id and client_secret and tenant_id:
        print("[auth] Using ClientSecretCredential (Service Principal – CI/CD mode)")
        return ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )

    if tenant_id:
        print(f"[auth] Using AzureCliCredential (local login, tenant: {tenant_id})")
        return AzureCliCredential(tenant_id=tenant_id)

    print("[auth] Using DefaultAzureCredential (Managed Identity / VS Code / etc.)")
    return DefaultAzureCredential()


# ──────────────────────────────────────────────────────────────────────────────
def get_ml_client(cfg: dict) -> MLClient:
    """Build and return an authenticated MLClient."""
    credential = get_credential(cfg)
    try:
        client = MLClient(
            credential=credential,
            subscription_id=cfg["subscription_id"],
            resource_group_name=cfg["resource_group"],
            workspace_name=cfg["workspace_name"],
        )
        # Trigger a lightweight call to verify connectivity
        _ = client.workspaces.get(cfg["workspace_name"])
        print(f"[auth] ✅ Connected to workspace: '{cfg['workspace_name']}'")
        return client
    except Exception as exc:
        print(f"[ERROR] Failed to connect to Azure ML workspace: {exc}")
        sys.exit(1)
