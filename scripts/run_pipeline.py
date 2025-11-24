# scripts/run_pipeline.py
from azure.identity import ClientSecretCredential  # use client secret auth for the runner
from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes
import os
import sys

# Read config from env (set by workflow)
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.environ.get("AZ_RESOURCE_GROUP", "MLOPS")
WORKSPACE = os.environ.get("AZ_WORKSPACE", "mlops")
CLIENT_ID = os.environ.get("AZURE_CLIENT_ID")
TENANT_ID = os.environ.get("AZURE_TENANT_ID")
CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET")

# Debug print (non-sensitive): show which values are present
print("===== DEBUG VALUES =====")
print("SUBSCRIPTION_ID =", "SET" if SUBSCRIPTION_ID else "MISSING")
print("RESOURCE_GROUP  =", RESOURCE_GROUP)
print("WORKSPACE       =", WORKSPACE)
print("CLIENT_ID       =", "SET" if CLIENT_ID else "MISSING")
print("TENANT_ID       =", "SET" if TENANT_ID else "MISSING")
print("CLIENT_SECRET   =", "SET" if CLIENT_SECRET else "MISSING")
print("=========================")

if not (SUBSCRIPTION_ID and CLIENT_ID and TENANT_ID and CLIENT_SECRET):
    print("One or more required env vars are missing. Exiting.")
    sys.exit(1)

# Authenticate using client secret
cred = ClientSecretCredential(tenant_id=TENANT_ID, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

ml_client = MLClient(
    credential=cred,
    subscription_id=SUBSCRIPTION_ID,
    resource_group=RESOURCE_GROUP,
    workspace_name=WORKSPACE,
)

print("Connected to workspace:", WORKSPACE)

# now load components (these must exist in this workspace)
preprocess_component = ml_client.components.get("preprocess_component", version="1")
train_component = ml_client.components.get("train_tune_component", version="1")
register_component = ml_client.components.get("register_model_component", version="1")

print("Loaded components successfully.")

@pipeline()
def pricing_pipeline(raw_data):
    preprocess = preprocess_component(input_data=raw_data)
    train = train_component(processed_data=preprocess.outputs.output_data)
    register = register_component(model_path=train.outputs.model_folder)
    return {
        "processed": preprocess.outputs.output_data,
        "model": train.outputs.model_folder
    }

pipeline_job = pricing_pipeline(
    raw_data=Input(type=AssetTypes.URI_FILE, path="azureml://datastores/workspaceblobstore/paths/used_cars.csv")
)

submitted_job = ml_client.jobs.create_or_update(pipeline_job)
print("Pipeline submitted. Job name:", submitted_job.name)
