from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes
import os

# Azure ML workspace details
SUBSCRIPTION_ID = "4327687e-5856-460d-9b72-ac75b8b1a3d2"
RESOURCE_GROUP = "MLOPS"
WORKSPACE = "mlops"

# Debug prints
print("===== DEBUG VALUES =====")
print("SUBSCRIPTION_ID =", SUBSCRIPTION_ID)
print("RESOURCE_GROUP  =", RESOURCE_GROUP)
print("WORKSPACE       =", WORKSPACE)

print("\nEnvironment variables visible to job:")
print("AZURE_SUBSCRIPTION_ID =", os.getenv("AZURE_SUBSCRIPTION_ID"))
print("AZURE_TENANT_ID       =", os.getenv("AZURE_TENANT_ID"))
print("AZURE_CLIENT_ID       =", os.getenv("AZURE_CLIENT_ID"))
print("AZURE_CLIENT_SECRET   =", "SET" if os.getenv("AZURE_CLIENT_SECRET") else "NOT SET")
print("=========================\n")

# Authenticate from GitHub Actions
cred = DefaultAzureCredential()

# Connect to Azure ML workspace
ml_client = MLClient(
    credential=cred,
    subscription_id=SUBSCRIPTION_ID,
    resource_group=RESOURCE_GROUP,
    workspace_name=WORKSPACE,
)

print("Connected to workspace:", WORKSPACE)

# Load registered components
preprocess_component = ml_client.components.get("preprocess_component", version="1")
train_component = ml_client.components.get("train_tune_component", version="1")
register_component = ml_client.components.get("register_model_component", version="1")

print("Loaded components successfully into the workspace.")

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
