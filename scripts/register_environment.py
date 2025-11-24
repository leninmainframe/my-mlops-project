# scripts/register_environment.py
import os
import sys
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient, load_environment

# Accept multiple possible env var names to avoid mismatch
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID") or os.environ.get("AZ_SUBSCRIPTION_ID")
RESOURCE_GROUP  = os.environ.get("AZ_RESOURCE_GROUP") or os.environ.get("AZURE_RESOURCE_GROUP") or os.environ.get("RESOURCE_GROUP")
WORKSPACE       = os.environ.get("AZ_WORKSPACE") or os.environ.get("AZURE_WORKSPACE") or os.environ.get("WORKSPACE")

CLIENT_ID       = os.environ.get("AZURE_CLIENT_ID") or os.environ.get("CLIENT_ID")
TENANT_ID       = os.environ.get("AZURE_TENANT_ID") or os.environ.get("TENANT_ID")
CLIENT_SECRET   = os.environ.get("AZURE_CLIENT_SECRET") or os.environ.get("CLIENT_SECRET")

# non-sensitive debug summary
print("===== DEBUG (non-sensitive) =====")
print("SUBSCRIPTION_ID =", "SET" if SUBSCRIPTION_ID else "MISSING")
print("RESOURCE_GROUP  =", RESOURCE_GROUP or "MISSING")
print("WORKSPACE       =", WORKSPACE or "MISSING")
print("CLIENT_ID       =", "SET" if CLIENT_ID else "MISSING")
print("TENANT_ID       =", "SET" if TENANT_ID else "MISSING")
print("CLIENT_SECRET   =", "SET" if CLIENT_SECRET else "MISSING")
print("=================================")

# fail early with helpful hint
if not (SUBSCRIPTION_ID and RESOURCE_GROUP and WORKSPACE):
    print("Missing one of SUBSCRIPTION_ID / RESOURCE_GROUP / WORKSPACE. Check your workflow 'env:' mapping.")
    sys.exit(1)

# if you are using client secret (recommended for register steps) ensure client creds are present
if not (CLIENT_ID and TENANT_ID and CLIENT_SECRET):
    print("Missing client secret credentials (AZURE_CLIENT_ID / AZURE_TENANT_ID / AZURE_CLIENT_SECRET).")
    print("If you prefer OIDC instead, switch to DefaultAzureCredential and grant the GitHub OIDC federated credential access.")
    sys.exit(1)

# create credential and client
cred = ClientSecretCredential(tenant_id=TENANT_ID, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

ml_client = MLClient(
    credential=cred,
    subscription_id=SUBSCRIPTION_ID,
    resource_group=RESOURCE_GROUP,
    workspace_name=WORKSPACE,
)

print("Registering environment...")

# load env file that is located in repo root (workflow working dir)
env = load_environment("environment.yaml")

created = ml_client.environments.create_or_update(env)
print("Environment registered:", created.name, "version:", created.version)
