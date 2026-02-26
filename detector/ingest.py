"""load preprocessed complaint data"""

import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "customer_id", "channel", "product", "text"]
OPTIONAL_COLUMNS = ["company", "state", "sub_product", "issue_type", "narrative"]


def load_data(csv_path):
    """load preprocessed CSV, validate schema, assign window labels.
    returns DataFrame with a 'window' column (monthly period).
    """
    df = pd.read_csv(csv_path, low_memory=False)

    #Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    #  Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    n_bad = df["timestamp"].isna().sum()
    if n_bad > 0:
        print(f"Warning: dropping {n_bad} rows with unparseable timestamps")
        df = df.dropna(subset=["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    #assign monthly window labels
    df["window"] = df["timestamp"].dt.to_period("M")

    #Log optional columns
    present = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    if present:
        print(f"Optional columns present: {present}")

    print(f"Loaded {len(df):,} complaints across {df['window'].nunique()} windows")
    print(f"  Date range: {df['timestamp'].min()} — {df['timestamp'].max()}")
    print(f"  Products: {df['product'].nunique()}, Channels: {df['channel'].nunique()}")

    return df
