from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes

import json

cred = DefaultAzureCredential()

ml_client = MLClient(
    cred,
    subscription_id = "4327687e-5856-460d-9b72-ac75b8b1a3d2",
    resource_group = "MLOPS",
    workspace_name = "mlops"
)

pipeline_job = ml_client.jobs.get("pricing_ml_pipeline")
print("Triggered pipeline:", pipeline_job.name)
