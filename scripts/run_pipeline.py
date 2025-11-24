# scripts/run_pipeline.py
import os, sys
# CHANGE 1: Import ClientSecretCredential instead of DefaultAzureCredential
from azure.identity import ClientSecretCredential 
from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes

# Read config from env (set by workflow)
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.environ.get("AZ_RESOURCE_GROUP")
WORKSPACE = os.environ.get("AZ_WORKSPACE")

# CHANGE 2: Read additional environment variables for explicit authentication
CLIENT_ID = os.environ.get("AZURE_CLIENT_ID") 
TENANT_ID = os.environ.get("AZURE_TENANT_ID")
CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET")

print("===== DEBUG VALUES =====")
print("SUBSCRIPTION_ID =", "SET" if SUBSCRIPTION_ID else "MISSING")
print("RESOURCE_GROUP  =", RESOURCE_GROUP)
print("WORKSPACE       =", WORKSPACE)
print("CLIENT_ID       =", "SET" if CLIENT_ID else "MISSING")
print("TENANT_ID       =", "SET" if TENANT_ID else "MISSING")
print("CLIENT_SECRET   =", "SET" if CLIENT_SECRET else "MISSING")
print("=========================")

# CHANGE 3: Update missing check to include all SP secrets
if not SUBSCRIPTION_ID or not RESOURCE_GROUP or not WORKSPACE or not CLIENT_ID or not TENANT_ID or not CLIENT_SECRET:
    print("One or more required env vars (SUB_ID, RG, WKSP, CLIENT_ID, TENANT_ID, or CLIENT_SECRET) are missing. Exiting.")
    sys.exit(1)

# CHANGE 4: Use ClientSecretCredential for explicit authentication
cred = ClientSecretCredential(
    tenant_id=TENANT_ID,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
)

# Initialize MLClient with the new explicit credential
ml_client = MLClient(
    cred, 
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP, 
    workspace_name=WORKSPACE
)
print("Connected to workspace:", WORKSPACE)

# Load components – these must exist in your workspace already
preprocess_component = ml_client.components.get("preprocess_component", version="1")
train_component = ml_client.components.get("train_tune_component", version="1")
register_component = ml_client.components.get("register_model_component", version="1")
print("Loaded components successfully.")

@pipeline()
def pricing_pipeline(raw_data):
    preprocess = preprocess_component(input_data=raw_data)
    train = train_component(processed_data=preprocess.outputs.output_data)
    register = register_component(model_path=train.outputs.model_folder)
    return {"processed": preprocess.outputs.output_data, "model": train.outputs.model_folder}

pipeline_job = pricing_pipeline(raw_data=Input(type=AssetTypes.URI_FILE,
                                                 path="azureml://datastores/workspaceblobstore/paths/used_cars.csv"))

submitted_job = ml_client.jobs.create_or_update(pipeline_job)
print("Pipeline submitted. Job name:", submitted_job.name)
