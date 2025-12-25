import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from datetime import datetime
from src.data.storage import fetch_data
from src.config import RAW_DATA_FILE, SNAPSHOTS_DIR

# Constants
REDDIT_TABLE = "reddit_mental_health_embeddings"
CONTROL_TABLE = "reddit_safe_embeddings"

def main():
    print("--- Starting Data Ingestion ---")
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    
    # 1. Fetch Data
    print(f"Fetching from {REDDIT_TABLE}...")
    df_risk = fetch_data(REDDIT_TABLE)
    df_risk['dataset_type'] = REDDIT_TABLE
    
    print(f"Fetching from {CONTROL_TABLE}...")
    df_control = fetch_data(CONTROL_TABLE)
    df_control['dataset_type'] = CONTROL_TABLE
    
    # 2. Combine
    print("Combining datasets...")
    df_all = pd.concat([df_risk, df_control], ignore_index=True)
    print(f"Total records: {len(df_all)}")
    
    # 3. Snapshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(SNAPSHOTS_DIR, f"raw_data_{timestamp}.pkl")
    
    print(f"Saving snapshot to {snapshot_path}...")
    # Use pickle to preserve numpy arrays and types
    df_all.to_pickle(snapshot_path)
    
    # 4. Update 'latest' link
    print(f"Updating {RAW_DATA_FILE}...")
    if os.path.exists(RAW_DATA_FILE):
        os.remove(RAW_DATA_FILE)
    
    # Copy/rewrite the file to be the latest
    df_all.to_pickle(RAW_DATA_FILE)
    
    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    main()
