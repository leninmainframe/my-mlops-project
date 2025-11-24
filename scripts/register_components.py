# scripts/register_components.py
import os
import sys
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient, load_component

# 1. Read config from env (set by workflow)
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.environ.get("AZ_RESOURCE_GROUP")
WORKSPACE = os.environ.get("AZ_WORKSPACE")
CLIENT_ID = os.environ.get("AZURE_CLIENT_ID") 
TENANT_ID = os.environ.get("AZURE_TENANT_ID")
CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET")

if not all([SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE, CLIENT_ID, TENANT_ID, CLIENT_SECRET]):
    print("One or more required environment variables are missing. Exiting.")
    sys.exit(1)

# 2. Authenticate using explicit client secret
cred = ClientSecretCredential(
    tenant_id=TENANT_ID,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
)

ml_client = MLClient(
    cred, 
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP, 
    workspace_name=WORKSPACE
)
print(f"Connected to workspace: {WORKSPACE}")

# 3. Register Components
component_files = [
    "./components/preprocess_component.yml",
    # Add your other component YAML files here, e.g.:
    # "./components/train_tune_component.yml",
    # "./components/register_model_component.yml",
]

print("Starting component registration...")
for file_path in component_files:
    try:
        component_def = load_component(source=file_path)
        ml_client.components.create_or_update(component_def) 
        print(f"Successfully registered: {component_def.name}")
    except Exception as e:
        print(f"Error registering {file_path}: {e}")
        sys.exit(1)

print("All components registered successfully.")
