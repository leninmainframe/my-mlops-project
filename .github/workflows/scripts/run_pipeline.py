from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes

import json

cred = DefaultAzureCredential()

ml_client = MLClient(
    cred,
    subscription_id = "<replace>",
    resource_group = "<replace>",
    workspace_name = "<replace>"
)

pipeline_job = ml_client.jobs.get("pricing_ml_pipeline")

print("Triggered pipeline:", pipeline_job.name)
