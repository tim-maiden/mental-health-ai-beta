import os
import sys
import json
import ast
import requests
import io
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.config import PROGRESS_FILE, BATCH_SIZE, SUPABASE_URL, SUPABASE_KEY
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
    """Converts string representation of list to numpy array. Optimized for speed."""
    if x is None:
        return None
        
    # If it's already a list (some drivers do this automatically)
    if isinstance(x, list):
        return np.array(x, dtype=np.float32)
        
    # Fast path: json.loads is ~10-50x faster than ast.literal_eval
    try:
        return np.array(json.loads(x), dtype=np.float32)
    except (TypeError, json.JSONDecodeError):
        # Fallback to slower/safer parser if JSON fails
        try:
            return np.array(ast.literal_eval(x), dtype=np.float32)
        except (ValueError, SyntaxError):
            return None

def fetch_data(table_name, columns=None):
    """
    Fetches data as CSV directly from Supabase.
    ~10x faster than JSON loop because it avoids Python-side JSON parsing.
    """
    print(f"Fetching {table_name} as CSV stream...")
    
    # Construct the URL
    url = f"{SUPABASE_URL}/rest/v1/{table_name}"
    
    # Select specific columns if requested
    params = {}
    if columns:
        params['select'] = ",".join(columns)
    else:
        params['select'] = "*"
        
    # Headers: Request CSV format
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "text/csv" # CRITICAL: Tells Supabase to send CSV
    }
    
    # Stream the request
    # This keeps memory usage low during the download
    with requests.get(url, headers=headers, params=params, stream=True) as r:
        r.raise_for_status()
        
        # Read the CSV directly from the raw socket stream
        # low_memory=False ensures pandas guesses types accurately for large files
        df = pd.read_csv(r.raw, low_memory=False)
        
    # --- Post-Processing ---
    
    # Fix Postgres Array format: "{0.1,0.2}" -> np.array([0.1, 0.2])
    # Supabase CSV exports arrays as strings like "{value,value}"
    if 'embedding' in df.columns:
        print("Processing embeddings...")
        
        # Optimized parser for Postgres array format
        def parse_pg_array(x):
            if isinstance(x, str) and x.startswith('{') and x.endswith('}'):
                # Strip braces and split by comma
                # This is much faster than ast.literal_eval or json.loads for this specific format
                try:
                    return np.fromstring(x[1:-1], sep=',', dtype=np.float32)
                except ValueError:
                    return None
            return None

        # Apply the parser
        df['embedding_vec'] = df['embedding'].apply(parse_pg_array)
        df = df.dropna(subset=['embedding_vec'])
        
    # Cleanup strings
    if 'subreddit' in df.columns:
        df['clean_subreddit'] = df['subreddit'].astype(str).str.strip()
    
    df['dataset_type'] = table_name
    
    print(f"Loaded {len(df)} rows.")
    return df
