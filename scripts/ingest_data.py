import os
import sys

# Add project root to sys.path to ensure module resolution when running from scripts directory.
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
    
    # Ensure embedding is stored as list of floats (not string)
    if 'embedding_vec' in df_chunk.columns:
        # storage.py already parsed it to numpy array, convert to list for Parquet
        df_chunk['embedding'] = df_chunk['embedding_vec'].apply(lambda x: x.tolist() if x is not None else None)
    
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
    
    # Define explicit schema to unify 'predicted_emotions' type (List<String>) across different source tables.
    schema = pa.schema([
        ('post_id', pa.string()),
        ('input', pa.string()),
        ('embedding', pa.list_(pa.float32())), # Store as list of floats
        ('dataset_type', pa.string()),
        ('subreddit', pa.string()),
        ('author', pa.string()),
        ('emotion_scores', pa.struct([('negative', pa.float64()), ('positive', pa.float64())])),
        ('predicted_emotions', pa.list_(pa.string())) # Force list<string> even if null
    ])
    
    # Write to parquet with lock
    table = pa.Table.from_pandas(final_df, schema=schema)
    
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

    # 3. Upload to Hugging Face Hub
    print(f"Uploading to Hugging Face Hub (tim-maiden/mental-health-ai)...")
    from datasets import Dataset

    try:
        # Load the parquet file we just wrote
        dataset = Dataset.from_parquet(local_filename)
        
        # Push to hub
        # private=True is recommended for Reddit data to avoid PII issues, change if desired.
        dataset.push_to_hub("tim-maiden/mental-health-ai", private=True)
        print("Successfully uploaded to Hugging Face Hub.")
        
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {e}")

    # Cleanup
    if os.path.exists(local_filename):
        os.remove(local_filename)
        
    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    main()
