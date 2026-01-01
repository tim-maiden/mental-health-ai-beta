import os
import sys
import time
import random
import json
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import concurrent.futures
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

def fetch_data(table_name, fetch_size=1000, columns=None, start_id=None, end_id=None):
    """Fetches data from a Supabase table using efficient cursor-based pagination.
    Supports optional start_id and end_id for parallel chunking.
    """
    if start_id is None:
        print(f"Fetching all data from {table_name}...")
    else:
        # Minimal logging for parallel workers
        pass 
    
    # If columns are not specified, default to this set
    if columns is None:
        columns = ["subreddit", "embedding", "input", "post_id", "chunk_id"]
    
    # We MUST fetch 'id' for cursor pagination to work reliably
    query_columns = list(set(columns + ["id"]))
    select_str = ", ".join(query_columns)
    
    all_data = []
    # Initialize cursor. If start_id provided, start just before it.
    last_id = (start_id - 1) if start_id is not None else 0
    total_fetched = 0
    
    while True:
        # Use ID-based cursor pagination (O(1) vs O(N) for offset)
        # Select rows where id > last_id
        query = supabase.table(table_name).select(select_str)
        
        # Order by ID to ensure we move forward correctly
        filter_builder = query.order("id", desc=False).gt("id", last_id)
        
        # Apply end_id cap if provided
        if end_id is not None:
            filter_builder = filter_builder.lte("id", end_id)
            
        # Retry loop with exponential backoff
        max_retries = 5
        response = None
        for attempt in range(max_retries):
            try:
                response = filter_builder.limit(fetch_size).execute()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"   -> Error fetching chunk (last_id={last_id}) after {max_retries} attempts: {e}")
                    raise e
                
                wait_time = (2 ** attempt) + random.random()
                time.sleep(wait_time)
        
        data = response.data
        if not data:
            if start_id is None:
                print(f"   -> No more data found (Last ID: {last_id}).")
            break
            
        all_data.extend(data)
        
        # Update cursor
        # Assumes 'id' is in the response and is sortable (int)
        last_id = data[-1]['id']
        total_fetched += len(data)
        
        # Print progress every ~10k rows
        if total_fetched % (fetch_size * 5) == 0:
             worker_tag = f"[Range start {start_id}]" if start_id else "[Single Thread]"
             print(f"   -> {worker_tag} Fetched {total_fetched} rows so far (Current ID: {last_id})...")

    # Only print this for single-threaded calls (fetch_data), parallel workers are silent to avoid spam
    if start_id is None:
        print(f"\nTotal rows from {table_name}: {len(all_data)}")
    
    df = pd.DataFrame(all_data)
    
    # Process embeddings
    if 'embedding' in df.columns and not df.empty:
        # Use a simpler message for parallel workers
        if start_id is None:
            print(f"Processing {len(df)} embeddings (converting from list/string to numpy)...")
        else:
            print(f"   -> [Range start {start_id}] Fetched all rows. Processing embeddings...", end='\r')

        # 1. Convert to list of values first to avoid pandas overhead in loop
        raw_values = df['embedding'].values
        processed_values = [None] * len(raw_values)
        
        chunk_size = 50000
        total = len(raw_values)
        
        for i in range(0, total, chunk_size):
            end = min(i + chunk_size, total)
            
            # Process this chunk
            chunk_res = []
            for val in raw_values[i:end]:
                chunk_res.append(process_embedding_str(val))
            
            # Assign back
            processed_values[i:end] = chunk_res

        df['embedding_vec'] = processed_values
        df = df.dropna(subset=['embedding_vec'])
    
    # Standardize subreddit names
    if 'subreddit' in df.columns:
        df['clean_subreddit'] = df['subreddit'].astype(str).str.strip()
        
    # Add a type label (Risk vs Control) for potential usage
    df['dataset_type'] = table_name
    
    return df

def get_id_range(table_name):
    """Helper to get min and max IDs for parallel fetching."""
    try:
        # Get min id
        res_min = supabase.table(table_name).select("id").order("id", desc=False).limit(1).execute()
        min_id = res_min.data[0]['id'] if res_min.data else 0
        
        # Get max id
        res_max = supabase.table(table_name).select("id").order("id", desc=True).limit(1).execute()
        max_id = res_max.data[0]['id'] if res_max.data else 0
        
        return min_id, max_id
    except Exception as e:
        print(f"Error getting ID range for {table_name}: {e}")
        return 0, 0

def fetch_data_parallel(table_name, columns=None, num_workers=30):
    """
    Fetches data from Supabase in parallel chunks.
    Faster for large datasets (e.g., >100k rows) by utilizing network bandwidth.
    """
    print(f"Starting parallel fetch for {table_name} with {num_workers} workers...")
    
    min_id, max_id = get_id_range(table_name)
    print(f"   -> ID Range: {min_id} to {max_id}")
    
    if max_id == 0 or min_id > max_id:
        print("   -> Table appears empty or inaccessible.")
        return pd.DataFrame()

    total_range = max_id - min_id
    if total_range < 1000:
        # Small table, just fetch normally
        return fetch_data(table_name, columns=columns)
        
    # Calculate ranges
    chunk_size = math.ceil(total_range / num_workers)
    ranges = []
    
    current_start = min_id
    for _ in range(num_workers):
        current_end = current_start + chunk_size
        # Ensure we don't go past max_id (though lte handles it, cleaner to be precise)
        actual_end = min(current_end, max_id)
        
        ranges.append((current_start, actual_end))
        current_start = actual_end + 1
        
        if current_start > max_id:
            break
            
    print(f"   -> Launching {len(ranges)} threads...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Map ranges to futures
        future_to_range = {
            executor.submit(fetch_data, table_name, 2000, columns, r[0], r[1]): r 
            for r in ranges
        }
        
        for future in concurrent.futures.as_completed(future_to_range):
            r_range = future_to_range[future]
            try:
                df_chunk = future.result()
                results.append(df_chunk)
                print(f"      -> Chunk {r_range} finished: {len(df_chunk)} rows")
            except Exception as exc:
                print(f"      -> Chunk {r_range} generated an exception: {exc}")

    if not results:
        return pd.DataFrame()
        
    print("   -> Concatenating results...")
    final_df = pd.concat(results, ignore_index=True)
    
    # Deduplicate just in case of overlap (shouldn't happen with strict ranges but safe)
    final_df = final_df.drop_duplicates(subset=['id'])
    
    print(f"Parallel fetch complete. Total rows: {len(final_df)}")
    return final_df
