# 🚗 Used-Car Price Prediction — Azure ML MLOps Pipeline

An end-to-end MLOps project that trains, tunes, and registers a **GradientBoosting regression model** for used-car price prediction on **Azure Machine Learning**, fully automated through **GitHub Actions CI/CD**.

---

## 📁 Project Structure

```
my-mlops-project/
│
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml          # GitHub Actions CI/CD workflow
│
├── components/                          # AzureML component YAML definitions
│   ├── preprocess_component.yml         # Component 1: Data Preprocessing
│   ├── train_tune_component.yml         # Component 2: Training & Tuning
│   └── register_model_component.yml     # Component 3: Model Registration
│
├── pipeline_components/                 # Executable Python scripts (run on AML compute)
│   ├── preprocess.py                    # Clean, engineer features, log stats to MLflow
│   ├── train_tune.py                    # Optuna HPO + MLflow nested run logging
│   └── register_model.py               # Register model in AML registry + MLflow
│
├── scripts/                             # Orchestration scripts (run locally / in CI)
│   ├── auth_helper.py                   # Shared credential & config helper
│   ├── register_environment.py          # Register pricing-env in AML
│   ├── register_components.py           # Register all 3 pipeline components in AML
│   └── execute_pipeline.py             # Submit the end-to-end pipeline job
│
├── config.json                          # Local dev config (subscription, workspace, etc.)
├── environment.yaml                     # AzureML environment definition (conda deps)
├── used_cars.csv                        # Raw dataset
└── README.md
```

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────┐     ┌──────────────────────────┐     ┌─────────────────────┐
│  preprocess.py      │     │  train_tune.py            │     │  register_model.py  │
│                     │     │                           │     │                     │
│  • Read raw CSV     │────▶│  • Load processed CSV     │────▶│  • Load artifact    │
│  • Strip units      │     │  • Optuna HPO (20 trials) │     │  • Register in AML  │
│  • Drop nulls       │     │  • MLflow nested runs     │     │  • Log to MLflow    │
│  • Engineer feats   │     │  • Train final GB model   │     │                     │
│  • Log to MLflow    │     │  • Save model.joblib      │     │  ✅ Model Registry  │
│                     │     │  • MLflow model registry  │     │                     │
└─────────────────────┘     └──────────────────────────┘     └─────────────────────┘
    ↓ processed_data.csv         ↓ model.joblib + metadata.json
```

---

## 🚀 Quick Start — Local Setup

### 1. Prerequisites

```bash
pip install azure-ai-ml azure-identity optuna scikit-learn mlflow pandas numpy joblib
```

### 2. Authenticate to Azure

```bash
# Service Principal login (recommended)
az login --service-principal \
  --username <CLIENT_ID> \
  --password <CLIENT_SECRET> \
  --tenant   <TENANT_ID>

# OR interactive login
az login --tenant <TENANT_ID>
```

### 3. Configure workspace

Edit `config.json`:
```json
{
  "subscription_id": "<YOUR_SUBSCRIPTION_ID>",
  "resource_group":  "<YOUR_RESOURCE_GROUP>",
  "workspace_name":  "<YOUR_WORKSPACE_NAME>",
  "tenant_id":       "<YOUR_TENANT_ID>",
  "compute":         "<YOUR_COMPUTE_CLUSTER>"
}
```

### 4. Run the pipeline

```bash
# Step 1: Register the conda environment (only needed once)
python scripts/register_environment.py

# Step 2: Register all 3 pipeline components (run after any component change)
python scripts/register_components.py

# Step 3: Submit the end-to-end pipeline
python scripts/execute_pipeline.py
```

---

## ⚙️ GitHub Actions CI/CD

### Required Secrets

Add these under **GitHub → Settings → Secrets → Actions**:

| Secret Name | Description |
|---|---|
| `AZURE_CLIENT_ID` | Service Principal Application (client) ID |
| `AZURE_CLIENT_SECRET` | Service Principal secret value |
| `AZURE_TENANT_ID` | Azure AD tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |
| `AZ_RESOURCE_GROUP` | Resource group containing the AML workspace |
| `AZ_WORKSPACE` | Azure ML workspace name |
| `AZ_COMPUTE` | Compute cluster name (e.g. `ml-cluster`) |

### Triggers

| Event | Effect |
|---|---|
| Push to `main` or `develop` (touching pipeline code) | Auto-runs full MLOps pipeline |
| Manual via GitHub UI (`workflow_dispatch`) | Run on demand |

---

## 🔄 MLflow Tracking

Every pipeline run logs the following to **AzureML's built-in MLflow tracking**:

| Script | What's logged |
|---|---|
| `preprocess.py` | `raw_row_count`, `clean_row_count`, `dropped_rows`, `feature_count`, per-column missing counts |
| `train_tune.py` | One **parent run** (`optuna_search`) + **20 nested runs** (one per trial) with `mae`, `rmse`, `r2`, hyperparams |
| `train_tune.py` | `final_model` run with best params + `model.joblib` artifact logged to MLflow Registry |
| `register_model.py` | `registered_model_name`, `registered_model_version`, performance metrics |

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `Kilometers_Driven` | Total distance driven (km) |
| `Mileage` | Fuel efficiency (kmpl) |
| `Engine` | Engine displacement (cc) |
| `Power` | Peak power output (bhp) |
| `Seats` | Seating capacity |
| `power_per_cc` | **Engineered**: Power / Engine — captures performance efficiency |
| `Segment_enc` | **Engineered**: 1 if Luxury segment, else 0 |

**Target**: `log_price` = log₁(price) — log-transform stabilises the skewed price distribution.

---

## 🧠 Model: GradientBoostingRegressor

Hyperparameters tuned via **Optuna** (20 trials, minimise MAE on original price scale):

| Hyperparameter | Search Range |
|---|---|
| `n_estimators` | 50 – 300 |
| `learning_rate` | 0.01 – 0.3 (log scale) |
| `max_depth` | 2 – 8 |
| `subsample` | 0.5 – 1.0 |
| `min_samples_split` | 2 – 20 |

---

## 💡 Actionable Insights & Recommendations

### Key Takeaways

1. **Feature Engineering drives performance** — `power_per_cc` (power-to-displacement ratio) is the single most informative derived feature; it captures the car's performance tier more precisely than raw Engine or Power alone. Always engineer domain-relevant features before HPO.

2. **Log-transform the target** — Used-car prices are heavily right-skewed. Training on `log(price)` instead of raw price stabilises gradients and significantly improves MAE on high-value vehicles.

3. **Optuna + MLflow = reproducible HPO** — Logging each Optuna trial as a nested MLflow run gives full traceability of the hyperparameter search. You can re-create any prior model version from the MLflow UI.

4. **Automate, don't manually trigger** — The CI/CD workflow auto-runs on every push that touches `pipeline_components/**`. This enforces that models are always retrained from a validated codebase, eliminating "works on my machine" issues.

5. **Pin component versions** — Components are versioned (v2) in AzureML. When you update a component, bump the version and update `COMPONENT_VERSIONS` in `execute_pipeline.py`. This gives you full rollback capability.

6. **Compute cluster autoscaling** — Set `min_instances=0` on the `ml-cluster` to pay only when a job is running. The pipeline will spin up nodes and scale back to zero automatically.

7. **Monitor for data drift** — Schedule a weekly drift-detection job using AzureML Data Monitor on the `workspaceblobstore/used_cars.csv` path. Retrain automatically when drift exceeds a threshold (e.g., Jensen-Shannon divergence > 0.1).

### Business Recommendations

- **Pricing API**: Expose the registered model as an AzureML **Managed Online Endpoint** for real-time price scoring in the dealer portal.
- **Retraining cadence**: Trigger the pipeline monthly or when > 5% new inventory data is added.
- **A/B testing**: Use AzureML's traffic-split deployment to gradually roll out new model versions before promoting to production.

---

## 🔗 Useful Links

- [Azure ML Documentation](https://learn.microsoft.com/azure/machine-learning/)
- [MLflow on Azure ML](https://learn.microsoft.com/azure/machine-learning/concept-mlflow)
- [Optuna Documentation](https://optuna.readthedocs.io/)
