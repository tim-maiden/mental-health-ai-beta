import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from datetime import datetime
from src.data.storage import fetch_data_parallel
from src.config import RAW_DATA_FILE, SNAPSHOTS_DIR, S3_BUCKET_NAME

# Constants
REDDIT_TABLE = "reddit_mental_health_embeddings"
CONTROL_TABLE = "reddit_safe_embeddings"

def main():
    print("--- Starting Data Ingestion ---")
    
    # 1. Fetch Data
    print(f"Fetching from {REDDIT_TABLE}...")
    # Added 'author' as required for train/test split
    df_risk = fetch_data_parallel(
        REDDIT_TABLE, 
        columns=["subreddit", "embedding", "input", "post_id", "author", "emotion_scores"],
        num_workers=5
    )
    df_risk['dataset_type'] = REDDIT_TABLE
    
    print(f"Fetching from {CONTROL_TABLE}...")
    # Added 'author'
    df_control = fetch_data_parallel(
        CONTROL_TABLE, 
        columns=["subreddit", "embedding", "input", "post_id", "author", "predicted_emotions", "emotion_scores"],
        num_workers=5
    )
    df_control['dataset_type'] = CONTROL_TABLE
    
    # 2. Combine
    print("Combining datasets...")
    df_all = pd.concat([df_risk, df_control], ignore_index=True)
    print(f"Total records: {len(df_all)}")
    
    # 3. Snapshot (Direct to S3)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # define S3 paths
    s3_path = f"s3://{S3_BUCKET_NAME}/data/snapshots/raw_data_{timestamp}.parquet"
    latest_path = RAW_DATA_FILE # This should now be the S3 URL from config

    print(f"Streaming snapshot to {s3_path}...")

    # Filter columns before saving to reduce RAM/Network usage
    keep_cols = [
        'post_id', 'input', 'embedding', 'dataset_type', 
        'subreddit', 'author', 'emotion_scores', 'predicted_emotions'
    ]
    # Only keep columns that actually exist in the dataframe
    final_cols = [c for c in keep_cols if c in df_all.columns]

    storage_options = {
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY")
    }

    # WRITE TO S3 DIRECTLY
    # engine='pyarrow' is fast and supports S3
    # compression='snappy' is a good balance of speed/size
    df_all[final_cols].to_parquet(
        s3_path, 
        engine='pyarrow', 
        compression='snappy',
        index=False,
        storage_options=storage_options
    )

    print(f"Updating latest link at {latest_path}...")
    # You can simply write the file again to the 'latest' path
    df_all[final_cols].to_parquet(
        latest_path,
        engine='pyarrow',
        compression='snappy',
        index=False,
        storage_options=storage_options
    )
    
    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    main()
