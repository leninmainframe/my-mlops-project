#!/usr/bin/env python
"""
register_model.py
- Registers model artifact into Azure ML model registry when run as an Azure ML job.
- Expects these environment variables (set by the job context):
  AZUREML_SERVICE_PRINCIPAL_CLIENT_ID etc OR uses DefaultAzureCredential
"""
import argparse, os, json
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to model artifact (folder or file)")
parser.add_argument("--model_name", type=str, default="used_cars_pricing_model", help="Name to register")
parser.add_argument("--subscription_id", type=str, default=os.environ.get("AZ_SUBSCRIPTION_ID"))
parser.add_argument("--resource_group", type=str, default=os.environ.get("AZ_RESOURCE_GROUP"))
parser.add_argument("--workspace", type=str, default=os.environ.get("AZ_WORKSPACE"))
args = parser.parse_args()

# Try to get credentials and register model
cred = DefaultAzureCredential()
if not args.subscription_id or not args.resource_group or not args.workspace:
    print(json.dumps({"error": "subscription_id, resource_group, workspace must be provided either as args or env vars"}))
else:
    ml_client = MLClient(cred, args.subscription_id, args.resource_group, args.workspace)
    # register model
    model = Model(path=args.model_path, name=args.model_name, description="Used cars pricing model")
    created = ml_client.models.create_or_update(model)
    print(json.dumps({"registered_model_name": created.name, "registered_model_version": created.version}))
