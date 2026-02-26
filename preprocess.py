"""CFPB complaints CSV --> standardization

Usage:
    python preprocess.py --input ~/complaints.csv --output data.csv --max-rows 100000
"""

import argparse
import sys

import pandas as pd
import numpy as np


# merge evolved CFPB product names into stable categories
PRODUCT_MAP = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit reporting": "Credit reporting",
    "Credit card or prepaid card": "Credit card",
    "Credit card": "Credit card",
    "Checking or savings account": "Bank account",
    "Bank account or service": "Bank account",
    "Student loan": "Student loan",
    "Federal student loan": "Student loan",
    "Private student loan": "Student loan",
    "Student loan servicing": "Student loan",
    "Payday loan, title loan, or personal loan": "Personal loan",
    "Payday loan, title loan, personal loan, or advance": "Personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Personal loan",
    "Payday loan": "Personal loan",
    "Consumer Loan": "Personal loan",
    "Vehicle loan or lease": "Vehicle loan",
    "Vehicle loan": "Vehicle loan",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Money transfers": "Money transfer",
    "Prepaid card": "Prepaid card",
    "Mortgage": "Mortgage",
    "Debt collection": "Debt collection",
    "Debt or credit management": "Debt collection",
}


def normalize_product(name):
    if pd.isna(name):
        return "Other"
    return PRODUCT_MAP.get(name, name)


def build_text(row):
    parts = []
    if pd.notna(row["Issue"]):
        parts.append(str(row["Issue"]))
    if pd.notna(row.get("Sub-issue")):
        parts.append(str(row["Sub-issue"]))
    return ": ".join(parts) if parts else ""


def preprocess(input_path, output_path, max_rows=100_000,
               start="2017-01-01", end="2026-02-01"):
    print(f"Reading {input_path} in chunks...")

    # first pass: read only date + ID columns to build sampling frame
    date_chunks = []
    chunk_size = 500_000
    for chunk in pd.read_csv(input_path, usecols=["Date received", "Complaint ID"],
                             chunksize=chunk_size, low_memory=False):
        chunk["Date received"] = pd.to_datetime(chunk["Date received"], format="mixed", errors="coerce")
        chunk = chunk.dropna(subset=["Date received"])
        chunk = chunk[(chunk["Date received"] >= start) & (chunk["Date received"] < end)]
        if len(chunk) > 0:
            chunk["month"] = chunk["Date received"].dt.to_period("M")
            date_chunks.append(chunk[["Complaint ID", "month"]])

    frame = pd.concat(date_chunks, ignore_index=True)
    total = len(frame)
    print(f"Total rows in date range: {total:,}")

    if total <= max_rows:
        sample_ids = set(frame["Complaint ID"])
        print(f"Using all {total:,} rows (under max_rows)")
    else:
        # stratified sample proportional per month
        month_counts = frame["month"].value_counts()
        month_fracs = month_counts / total
        sampled = []
        for month, frac in month_fracs.items():
            month_rows = frame[frame["month"] == month]
            n = max(1, int(round(frac * max_rows)))
            n = min(n, len(month_rows))
            sampled.append(month_rows.sample(n=n, random_state=42))
        sampled_frame = pd.concat(sampled, ignore_index=True)
        # Trim to exact max_rows if rounding pushed us over
        if len(sampled_frame) > max_rows:
            sampled_frame = sampled_frame.sample(n=max_rows, random_state=42)
        sample_ids = set(sampled_frame["Complaint ID"])
        print(f"Sampled {len(sample_ids):,} rows (stratified by month)")

    del frame

    # second pass: read full data for sampled IDs
    print("Second pass: reading full rows for sample...")
    result_chunks = []
    for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
        matches = chunk[chunk["Complaint ID"].isin(sample_ids)]
        if len(matches) > 0:
            result_chunks.append(matches)

    df = pd.concat(result_chunks, ignore_index=True)
    print(f"Loaded {len(df):,} rows")

    # map columns
    df["timestamp"] = pd.to_datetime(df["Date received"], format="mixed", errors="coerce")
    df["customer_id"] = df["Complaint ID"].astype(str)
    df["channel"] = df["Submitted via"].fillna("Unknown")
    df["product"] = df["Product"].apply(normalize_product)
    df["text"] = df.apply(build_text, axis=1)

    # drop rows with empty text
    df = df[df["text"].str.len() > 0]

    # Optional columns to preserve
    optional_cols = {}
    for raw_col, out_col in [("Company", "company"), ("State", "state"),
                              ("Sub-product", "sub_product"), ("Issue", "issue_type"),
                              ("Consumer complaint narrative", "narrative")]:
        if raw_col in df.columns:
            optional_cols[out_col] = df[raw_col]

    #build output
    out = df[["timestamp", "customer_id", "channel", "product", "text"]].copy()
    for col, series in optional_cols.items():
        out[col] = series.values

    out = out.sort_values("timestamp").reset_index(drop=True)

    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out):,} rows to {output_path}")

    # Summary stats
    print(f"\nDate range: {out['timestamp'].min()} — {out['timestamp'].max()}")
    print(f"Products: {out['product'].nunique()}")
    print(f"Channels: {out['channel'].nunique()}")
    months = pd.to_datetime(out["timestamp"]).dt.to_period("M")
    print(f"Months: {months.nunique()}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess CFPB complaints for Conway Detector")
    parser.add_argument("--input", required=True, help="Path to raw CFPB complaints CSV")
    parser.add_argument("--output", default="data.csv", help="Output CSV path")
    parser.add_argument("--max-rows", type=int, default=100_000, help="Max rows in sample")
    parser.add_argument("--start", default="2017-01-01", help="Start date (inclusive)")
    parser.add_argument("--end", default="2026-02-01", help="End date (exclusive)")
    args = parser.parse_args()
    preprocess(args.input, args.output, args.max_rows, args.start, args.end)


if __name__ == "__main__":
    main()
