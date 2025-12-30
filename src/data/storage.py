import os
import sys
import json
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.config import PROGRESS_FILE, BATCH_SIZE
from src.core.clients import supabase
from src.services.embeddings import embed_dataframe

def load_progress():
    """Loads the progress from the progress file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_progress(progress_data):
    """Saves the progress to the progress file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress_data, f, indent=4)

def embed_and_upload_dataframe_in_batches(df: pd.DataFrame, table_name: str, batch_size: int = BATCH_SIZE, int_columns: list = []):
    """Embeds a DataFrame and uploads it to Supabase in batches, with resume capability."""
    progress = load_progress()
    rows_to_skip = progress.get(table_name, 0)

    if rows_to_skip > 0:
        print(f"Resuming upload for {table_name} from row {rows_to_skip}.")
        df_remaining = df.iloc[rows_to_skip:].copy()
    else:
        df_remaining = df.copy()

    # Ensure integer columns have the correct type, allowing for NaNs
    for col in int_columns:
        if col in df_remaining.columns:
            df_remaining.loc[:, col] = pd.to_numeric(df_remaining[col], errors='coerce').astype('Int64')

    for i in tqdm(range(0, len(df_remaining), batch_size), desc=f"Uploading to {table_name}"):
        batch_df = df_remaining.iloc[i:i+batch_size].copy()
        
        # --- Data Cleaning ---
        # Clean null characters from all object (likely string) columns
        # BUT skip columns that are actually lists (like 'emotions')
        for col in batch_df.select_dtypes(include=['object']).columns:
            # Check if the column likely contains lists (by checking first non-null element)
            first_valid_index = batch_df[col].first_valid_index()
            if first_valid_index is not None:
                sample_val = batch_df.loc[first_valid_index, col]
                if isinstance(sample_val, list):
                    continue

            batch_df.loc[:, col] = batch_df[col].astype(str).str.replace('\u0000', '', regex=False)
        
        # Calculate batch number relative to the start of the whole dataset
        current_row_index = rows_to_skip + i
        batch_number = current_row_index // batch_size + 1

        # Embed the batch
        embedded_batch_df = embed_dataframe(batch_df, desc=f"Embedding batch {batch_number}")
        
        # Prepare data for Supabase
        # Replace pandas' NA with None and also handle numpy's nan
        embedded_batch_df = embedded_batch_df.replace({np.nan: None, pd.NA: None})
        records_to_insert = embedded_batch_df.to_dict(orient='records')

        # Filter out records with missing embeddings
        records_to_insert = [r for r in records_to_insert if r.get('embedding') is not None]

        if not records_to_insert:
            print(f"Skipping batch {batch_number} as no embeddings were generated.")
            continue

        try:
            supabase.table(table_name).insert(records_to_insert).execute()
            # Update and save progress after a successful batch upload
            progress[table_name] = current_row_index + len(batch_df)
            save_progress(progress)
        except Exception as e:
            print(f"Error inserting batch {batch_number} into {table_name}. Error: {e}")
            # Stop execution on error to avoid inconsistent progress
            print("Exiting script to prevent further errors. Please check the error and restart.")
            sys.exit(1)

def process_embedding_str(x):
    """Converts string representation of list to numpy array."""
    if x is None:
        return None
    try:
        if isinstance(x, list):
            return np.array(x, dtype=np.float32)
        return np.array(ast.literal_eval(x), dtype=np.float32)
    except (ValueError, SyntaxError):
        return None

def fetch_data(table_name, fetch_size=1000, columns=None):
    """Fetches all data from a Supabase table."""
    print(f"Fetching data from {table_name}...")
    if columns is None:
        columns = ["subreddit", "embedding", "input", "post_id", "chunk_id"]
    
    select_str = ", ".join(columns)
    all_data = []
    page = 0
    
    while True:
        offset = page * fetch_size
        # Select specified columns
        query = supabase.table(table_name).select(select_str)
        
        # Ensure deterministic ordering for pagination
        if "post_id" in columns and "chunk_id" in columns:
            query = query.order("post_id", desc=False).order("chunk_id", desc=False)
            
        response = query.limit(fetch_size).offset(offset).execute()
        data = response.data
        if not data:
            break
        all_data.extend(data)
        page += 1
        print(f"   -> Fetched {len(all_data)} rows...", end="\r")
    
    print(f"\nTotal rows from {table_name}: {len(all_data)}")
    
    df = pd.DataFrame(all_data)
    df['embedding_vec'] = df['embedding'].apply(process_embedding_str)
    df = df.dropna(subset=['embedding_vec'])
    
    # Standardize subreddit names
    df['clean_subreddit'] = df['subreddit'].astype(str).str.strip()
    # Add a type label (Risk vs Control) for potential usage
    df['dataset_type'] = table_name
    
    return df
