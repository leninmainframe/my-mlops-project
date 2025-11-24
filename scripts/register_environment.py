import os
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient, load_environment

subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
resource_group  = os.environ["AZ_RESOURCE_GROUP"]
workspace       = os.environ["AZ_WORKSPACE"]
client_id       = os.environ["AZURE_CLIENT_ID"]
tenant_id       = os.environ["AZURE_TENANT_ID"]
client_secret   = os.environ["AZURE_CLIENT_SECRET"]

cred = ClientSecretCredential(
    tenant_id=tenant_id,
    client_id=client_id,
    client_secret=client_secret
)

ml_client = MLClient(
    credential=cred,
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace
)

print("Registering environment...")

env = load_environment("environment.yaml")
created = ml_client.environments.create_or_update(env)

print("Environment registered:", created.name, "version:", created.version)
