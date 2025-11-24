# scripts/register_components.py
import os
import sys
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_component

# --- Configuration & Authentication Setup ---

# 1. Read config from env (set by workflow YAML)
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.environ.get("AZ_RESOURCE_GROUP")
WORKSPACE = os.environ.get("AZ_WORKSPACE")

if not all([SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE]):
    print("One or more required environment variables (SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE) are missing. Exiting.")
    sys.exit(1)

# 2. Authenticate using DefaultAzureCredential
# This uses the token established by the 'Azure Login (OIDC)' step in the GitHub Actions workflow.
try:
    cred = DefaultAzureCredential()
    ml_client = MLClient(
        cred, 
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP, 
        workspace_name=WORKSPACE
    )
except Exception as e:
    print(f"Error during MLClient initialization: {e}")
    sys.exit(1)

print(f"Connected to workspace: {WORKSPACE}")

# --- Component Registration Logic ---

# List of component YAML files to register
# Ensure these paths are correct relative to where the script is run (usually the repo root)
component_files = [
    "./components/preprocess_component.yml",
    "./components/train_tune_component.yml",
    "./components/register_model_component.yml",
]

print("Starting component registration...")
for file_path in component_files:
    try:
        component_def = load_component(source=file_path)
        # Using create_or_update to register the component or update if it already exists
        ml_client.components.create_or_update(component_def) 
        print(f"Successfully registered: {component_def.name} (Version {component_def.version})")
    except Exception as e:
        print(f"Error registering {file_path}: {e}")
        # Exit to fail the workflow if registration fails
        sys.exit(1)

print("All components registered successfully.")
