# scripts/register_environment.py
import os
import sys
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_environment

# --- Configuration & Authentication Setup ---

# 1. Read config from env (set by workflow YAML)
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.environ.get("AZ_RESOURCE_GROUP")
WORKSPACE = os.environ.get("AZ_WORKSPACE")

if not all([SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE]):
    print("One or more required environment variables (SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE) are missing. Exiting.")
    sys.exit(1)

# 2. Authenticate using DefaultAzureCredential
# This uses the token established by the 'Azure Login (OIDC)' step.
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

# --- Environment Registration Logic ---

# Assuming your environment definition file is named environment.yaml and is in the root directory
# If you used 'pricing-env' previously, ensure that name matches here or in the environment.yaml file.
ENVIRONMENT_YAML_PATH = "environment.yaml" # or "environment.yml" if that's the name you used

if not os.path.exists(ENVIRONMENT_YAML_PATH):
    print(f"Error: Environment YAML file not found at {ENVIRONMENT_YAML_PATH}. Exiting.")
    sys.exit(1)

print(f"Loading environment definition from {ENVIRONMENT_YAML_PATH}...")
try:
    env_def = load_environment(source=ENVIRONMENT_YAML_PATH)
    # Using create_or_update to register the environment or update its version
    created_env = ml_client.environments.create_or_update(env_def) 
    print(f"Successfully registered environment: {created_env.name} (Version {created_env.version})")
except Exception as e:
    print(f"Error registering environment: {e}")
    sys.exit(1)
