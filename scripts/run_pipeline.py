import json
from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes

# Load Azure config from config.json in repo
with open("config.json", "r") as f:
    cfg = json.load(f)

SUBSCRIPTION_ID = cfg["4327687e-5856-460d-9b72-ac75b8b1a3d2"]
RESOURCE_GROUP = cfg["MLOPS"]
WORKSPACE_NAME = cfg["mlops"]

print("🔄 Authenticating with Azure...")
cred = DefaultAzureCredential()

ml_client = MLClient(
    credential=cred,
    subscription_id=SUBSCRIPTION_ID,
    resource_group=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME
)

print("🚀 Submitting pipeline job...")

# Your raw CSV is in datastore as: azureml://datastores/workspaceblobstore/paths/used_cars.csv
raw_data_input = Input(
    type=AssetTypes.URI_FILE,
    path="azureml://datastores/workspaceblobstore/paths/used_cars.csv"
)

# Import your existing pipeline function compiled inside AzureML
pipeline_job = ml_client.jobs.create_or_update(
    {
        "name": "pricing_mlops_pipeline_ci_cd",
        "display_name": "Pricing Pipeline Triggered From GitHub",
        "jobs": {},
        "inputs": {
            "raw_data_input": raw_data_input
        },
        "type": "pipeline",
        "component": "pricing_ml_pipeline:1"   # pipeline name and version
    }
)

print("✅ Pipeline submitted successfully!")
print("🔗 Azure ML Studio URL:", pipeline_job.studio_url)
