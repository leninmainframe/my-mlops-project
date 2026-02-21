#!/usr/bin/env python
"""
preprocess.py
=============
Component 1 — Data Preprocessing
----------------------------------
• Reads raw CSV from the AzureML input path (file or folder)
• Performs cleaning: strips whitespace, coerces numerics, drops nulls
• Engineers features: power_per_cc, log_price, Segment_enc
• Logs dataset statistics to MLflow for observability
• Writes processed CSV to the AzureML output folder

Args (injected by AzureML component YAML):
    --input_data   : URI_FILE  – path to raw used_cars.csv
    --output_data  : URI_FOLDER – destination for processed_data.csv
"""

import argparse
import json
import os

import mlflow
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Preprocess raw used-car CSV data")
parser.add_argument("--input_data",  type=str, required=True,
                    help="Path to raw input CSV file or folder")
parser.add_argument("--output_data", type=str, required=True,
                    help="Path to output folder for processed CSV")
args = parser.parse_args()

os.makedirs(args.output_data, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data (support both URI_FILE and URI_FOLDER inputs)
# ---------------------------------------------------------------------------
input_path = args.input_data
if os.path.isdir(input_path):
    csv_files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
    if not csv_files:
        raise SystemExit(f"[preprocess] ERROR: No CSV files found in folder: {input_path}")
    input_path = os.path.join(input_path, csv_files[0])

print(f"[preprocess] Reading data from: {input_path}")
df = pd.read_csv(input_path)
raw_rows = len(df)
print(f"[preprocess] Raw dataset shape: {df.shape}")

# ---------------------------------------------------------------------------
# 2. Cleaning — strip column names, coerce numeric columns
# ---------------------------------------------------------------------------
df.columns = [c.strip() for c in df.columns]

NUMERIC_COLS = ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats", "price"]
for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)  # strip units (e.g. "bhp", "kmpl")
            .replace("", np.nan)
            .astype(float)
        )

# Drop rows with missing primary features or target
REQUIRED_COLS = ["price", "Kilometers_Driven", "Mileage", "Engine", "Power"]
missing_before = df[REQUIRED_COLS].isnull().sum().to_dict()
df = df.dropna(subset=REQUIRED_COLS)
clean_rows = len(df)
print(f"[preprocess] Rows after dropping nulls: {clean_rows} (dropped {raw_rows - clean_rows})")

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
if "Engine" in df.columns and "Power" in df.columns:
    df["power_per_cc"] = df["Power"] / (df["Engine"] + 1e-6)   # power-to-displacement ratio

df["log_price"] = np.log1p(df["price"])                        # log-transformed target

if "Segment" in df.columns:
    df["Segment_enc"] = df["Segment"].apply(
        lambda x: 1 if str(x).lower().strip() == "luxury" else 0
    )
else:
    df["Segment_enc"] = 0                                       # default if column absent

engineered_features = ["power_per_cc", "log_price", "Segment_enc"]
print(f"[preprocess] Engineered features: {engineered_features}")

# ---------------------------------------------------------------------------
# 4. MLflow logging — track dataset statistics for reproducibility
# ---------------------------------------------------------------------------
mlflow.start_run()
mlflow.log_metric("raw_row_count",   raw_rows)
mlflow.log_metric("clean_row_count", clean_rows)
mlflow.log_metric("dropped_rows",    raw_rows - clean_rows)
mlflow.log_metric("feature_count",   len(df.columns))
for col, n_missing in missing_before.items():
    mlflow.log_metric(f"missing_{col}", int(n_missing))
mlflow.log_param("numeric_cols_cleaned", json.dumps(NUMERIC_COLS))
mlflow.log_param("required_cols",        json.dumps(REQUIRED_COLS))
mlflow.end_run()
print("[preprocess] MLflow metrics logged.")

# ---------------------------------------------------------------------------
# 5. Save processed data
# ---------------------------------------------------------------------------
out_file = os.path.join(args.output_data, "processed_data.csv")
df.to_csv(out_file, index=False)

summary = {
    "processed_rows":   clean_rows,
    "columns":          list(df.columns),
    "output_file":      out_file,
}
print(json.dumps(summary, indent=2))
print("[preprocess] ✅ Done.")
