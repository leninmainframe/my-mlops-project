#!/usr/bin/env python
"""
preprocess.py
- Reads raw CSV from the input folder or file
- Performs light cleaning and feature engineering
- Saves processed CSV to output folder
"""
import argparse, os, pandas as pd, numpy as np, json
parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str, required=True, help="Input CSV file or folder")
parser.add_argument("--output_data", type=str, required=True, help="Output CSV folder")
args = parser.parse_args()

os.makedirs(args.output_data, exist_ok=True)

# Read CSV (support folder or file)
input_path = args.input_data
if os.path.isdir(input_path):
    files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
    if not files:
        raise SystemExit("No CSV found in input folder")
    df = pd.read_csv(os.path.join(input_path, files[0]))
else:
    df = pd.read_csv(input_path)

# Quick cleaning / conversions
df.columns = [c.strip() for c in df.columns]

for col in ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats", "price"]:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(r"[^0-9\\.-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

# Drop rows missing primary features
df = df.dropna(subset=["price", "Kilometers_Driven", "Mileage", "Engine", "Power"])

# Feature engineering
if "Engine" in df.columns and "Power" in df.columns:
    df["power_per_cc"] = df["Power"] / (df["Engine"] + 1e-6)
df["log_price"] = np.log1p(df["price"])
if "Segment" in df.columns:
    df["Segment_enc"] = df["Segment"].apply(lambda x: 1 if str(x).lower().strip()=="luxury" else 0)

# Save processed CSV
out_file = os.path.join(args.output_data, "processed_data.csv")
df.to_csv(out_file, index=False)
print(json.dumps({"processed_rows": len(df), "output_file": out_file}))
