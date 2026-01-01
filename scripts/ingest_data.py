import os
import sys

# CRITICAL: Add the project root (..) to sys.path BEFORE importing any src modules.
# This ensures that when running `python scripts/ingest_data.py` (where CWD is project root)
# OR `python ingest_data.py` (where CWD is scripts/), python can always find 'src'.
# We get the absolute path of the directory containing this script, then go up one level.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import threading
from datetime import datetime
from src.data.storage import fetch_data_parallel
from src.config import RAW_DATA_FILE, SNAPSHOTS_DIR, S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

# Constants
REDDIT_TABLE = "reddit_mental_health_embeddings"
CONTROL_TABLE = "reddit_safe_embeddings"

# Global writer state
writer = None
write_lock = threading.Lock()
local_filename = f"raw_data_temp.parquet"

def get_storage_options():
    return {
        "key": AWS_ACCESS_KEY_ID,
        "secret": AWS_SECRET_ACCESS_KEY,
        "client_kwargs": {"region_name": AWS_REGION}
    }

def process_chunk(df_chunk, dataset_type):
    """
    Callback to process and write a chunk of data to the local Parquet file.
    """
    global writer
    
    if df_chunk.empty:
        return

    # Add dataset type
    df_chunk['dataset_type'] = dataset_type
    
    # Filter columns
    keep_cols = [
        'post_id', 'input', 'embedding', 'dataset_type', 
        'subreddit', 'author', 'emotion_scores', 'predicted_emotions'
    ]
    # Ensure columns exist
    for col in keep_cols:
        if col not in df_chunk.columns:
            df_chunk[col] = None
            
    final_df = df_chunk[keep_cols]
    
    # Write to parquet with lock
    table = pa.Table.from_pandas(final_df)
    
    with write_lock:
        if writer is None:
            print(f"   -> Initializing Parquet Writer (Schema: {table.schema})...")
            writer = pq.ParquetWriter(local_filename, table.schema, compression='snappy')
        
        writer.write_table(table)

def main():
    print("--- Starting Streamed Data Ingestion ---")
    
    # Clean up previous temp file
    if os.path.exists(local_filename):
        os.remove(local_filename)

    try:
        # 1. Fetch & Write Risk Data
        print(f"Streaming from {REDDIT_TABLE}...")
        fetch_data_parallel(
            REDDIT_TABLE, 
            columns=["subreddit", "embedding", "input", "post_id", "author", "emotion_scores"],
            num_workers=5,
            chunk_callback=lambda df: process_chunk(df, REDDIT_TABLE)
        )
        
        # 2. Fetch & Write Control Data
        print(f"Streaming from {CONTROL_TABLE}...")
        fetch_data_parallel(
            CONTROL_TABLE, 
            columns=["subreddit", "embedding", "input", "post_id", "author", "predicted_emotions", "emotion_scores"],
            num_workers=5,
            chunk_callback=lambda df: process_chunk(df, CONTROL_TABLE)
        )
        
    finally:
        # Ensure writer is closed
        if writer:
            writer.close()
            print(f"Finished writing to local file: {local_filename}")
        else:
            print("No data was written.")
            return

    # 3. Upload to S3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_path = f"s3://{S3_BUCKET_NAME}/data/snapshots/raw_data_{timestamp}.parquet"
    latest_path = RAW_DATA_FILE

    print(f"Uploading to S3: {s3_path}...")
    
    # Use pandas to read back (efficiently) and write to S3? 
    # No, that defeats the purpose. Use boto3 or s3fs to upload the file directly.
    import s3fs
    fs = s3fs.S3FileSystem(key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY, client_kwargs={'region_name': AWS_REGION})
    
    # Upload to snapshot path
    fs.put(local_filename, s3_path)
    
    # Upload to latest path
    print(f"Updating latest link at {latest_path}...")
    fs.put(local_filename, latest_path)
    
    # Cleanup
    if os.path.exists(local_filename):
        os.remove(local_filename)
        
    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    main()
