import ast
import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.metrics.pairwise import cosine_similarity

# Add the project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.clients import supabase
from src.config import SUPABASE_URL, SUPABASE_KEY

OUTPUT_FILE = "lmsys_training_data.jsonl"
LMSYS_TABLE = "lmsys_chat_embeddings"
REDDIT_TABLE = "reddit_mental_health_embeddings"
BATCH_SIZE = 1000
# Set MAX_ROWS to None to process EVERYTHING, or an integer (e.g., 50000) for a test run
MAX_ROWS = None 

# --- 2. DATA PROCESSING HELPERS ---
def process_embedding_str(x):
    """Converts string representation of list to numpy array."""
    if x is None:
        return None
    try:
        # Fast path for standard lists
        if isinstance(x, list):
            return np.array(x, dtype=np.float32)
        # Parse string format
        return np.array(ast.literal_eval(x), dtype=np.float32)
    except (ValueError, SyntaxError):
        return None

def clean_label(val):
    """Standardizes the 'Cause' labels."""
    if pd.isna(val) or str(val).lower() == 'nan':
        return None
    
    val = str(val).lower().strip()
    if 'drug' in val: return 'Substance Use'
    elif 'personality' in val: return 'Personality'
    elif 'early' in val: return 'Early Life'
    elif 'trauma' in val: return 'Trauma & Stress'
    return val.title()

# --- 3. BUILD THE TEACHER INDEX ---
def load_and_build_index(supabase_client):
    print(f"[{datetime.now()}] Fetching Reddit Signal Data...")
    
    all_data = []
    page = 0
    fetch_size = 500
    
    # Fetch Loop
    while True:
        offset = page * fetch_size
        response = supabase_client.table(REDDIT_TABLE).select("*").limit(fetch_size).offset(offset).execute()
        data = response.data
        if not data:
            break
        all_data.extend(data)
        page += 1
        print(f"   -> Fetched {len(all_data)} rows...", end="\r")
    
    print(f"\n[{datetime.now()}] Total Signal Rows: {len(all_data)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Process Embeddings
    print("   -> Processing embeddings...")
    df['embedding_vec'] = df['embedding'].apply(process_embedding_str)
    df = df.dropna(subset=['embedding_vec'])
    
    # Clean Subreddits
    df = df[df['subreddit'].astype(str).str.lower() != 'nan']
    df['clean_subreddit'] = df['subreddit'].str.strip()
    
    # Clean Labels
    df['clean_label'] = df['label'].apply(clean_label)
    
    # Build Dictionary of Matrices
    print(f"[{datetime.now()}] Building Vector Index...")
    teacher_index = {}
    
    # Context Heads (Subreddits)
    # 1. Reddit Mental Health Subreddits (Risk Data)
    for sub in df['clean_subreddit'].unique():
        subset = df[df['clean_subreddit'] == sub]
        matrix = np.stack(subset['embedding_vec'].values)
        teacher_index[f"context_{sub}"] = matrix
        print(f"   -> Index 'context_{sub}': {matrix.shape[0]} vectors")

    # 2. Reddit Control Subreddits (Safe Data)
    # We need to fetch from the new table 'reddit_safe_embeddings'
    print(f"[{datetime.now()}] Fetching Reddit Control Data...")
    page = 0
    all_control_data = []
    
    while True:
        offset = page * fetch_size
        response = supabase_client.table("reddit_safe_embeddings").select("*").limit(fetch_size).offset(offset).execute()
        data = response.data
        if not data:
            break
        all_control_data.extend(data)
        page += 1
        print(f"   -> Fetched {len(all_control_data)} control rows...", end="\r")

    print(f"\n[{datetime.now()}] Total Control Rows: {len(all_control_data)}")

    # Process Control Data
    df_control = pd.DataFrame(all_control_data)
    df_control['embedding_vec'] = df_control['embedding'].apply(process_embedding_str)
    df_control = df_control.dropna(subset=['embedding_vec'])
    df_control['clean_subreddit'] = df_control['subreddit'].str.strip()

    # Add Control Subreddits to Index
    for sub in df_control['clean_subreddit'].unique():
        subset = df_control[df_control['clean_subreddit'] == sub]
        matrix = np.stack(subset['embedding_vec'].values)
        teacher_index[f"context_{sub}"] = matrix
        print(f"   -> Index 'context_{sub}': {matrix.shape[0]} vectors")
        
    return teacher_index

# --- 4. SCORING LOGIC ---
def calculate_risk_profile(chat_matrix, teacher_index):
    """
    Computes the max similarity of each chat against every Reddit category.
    Returns a list of dictionaries.
    """
    batch_size = chat_matrix.shape[0]
    profiles = [{} for _ in range(batch_size)]
    
    for category, ref_matrix in teacher_index.items():
        # Compute Cosine Similarity (Dot Product)
        # Result Shape: (Batch_Size, Reddit_Category_Size)
        sim_scores = cosine_similarity(chat_matrix, ref_matrix)
        
        # Max Pooling: Get the single best match score for this category
        max_scores = sim_scores.max(axis=1)
        
        # Store result
        for i in range(batch_size):
            profiles[i][category] = float(max_scores[i])
            
    return profiles

# --- 5. MAIN EXECUTION LOOP ---
def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: Missing Supabase credentials in environment variables")
        sys.exit(1)
    
    # 1. Build the Teacher
    teacher_index = load_and_build_index(supabase)
    
    print(f"\n[{datetime.now()}] Starting LMSYS Processing Loop...")
    print(f"   -> Output File: {OUTPUT_FILE}")
    print(f"   -> Language Filter: English")
    
    processed_count = 0
    total_processed = 0
    
    # Check if file exists to determine write mode
    file_mode = 'a' if os.path.exists(OUTPUT_FILE) else 'w'
    
    with open(OUTPUT_FILE, file_mode) as f_out:
        while True:
            # Check Exit Condition
            if MAX_ROWS and total_processed >= MAX_ROWS:
                print("\nReached MAX_ROWS limit.")
                break
                
            # A. Fetch Batch
            try:
                response = supabase.table(LMSYS_TABLE)\
                    .select("id, input, embedding")\
                    .eq("language", "English")\
                    .limit(BATCH_SIZE)\
                    .offset(processed_count)\
                    .execute()
            except Exception as e:
                print(f"\nError fetching batch: {e}")
                break
                
            batch_data = response.data
            
            # Stop if no more data
            if not batch_data:
                print("\nNo more data found in Supabase.")
                break
            
            # Update offset for next iteration (Supabase requires absolute offset)
            processed_count += len(batch_data)
            
            # B. Filter & Validate Batch
            valid_rows = []
            valid_vectors = []
            
            for row in batch_data:
                input_text = row.get('input')
                if not input_text: continue
                
                vec = process_embedding_str(row.get('embedding'))
                if vec is not None:
                    valid_rows.append(row)
                    valid_vectors.append(vec)
            
            if not valid_vectors:
                print(f"   -> Skipped empty/invalid batch... (Offset: {processed_count})", end="\r")
                continue
                
            # C. Compute Risks
            chat_matrix = np.stack(valid_vectors)
            risk_profiles = calculate_risk_profile(chat_matrix, teacher_index)
            
            # D. Write to Disk
            for i, row in enumerate(valid_rows):
                record = {
                    "lmsys_id": row['id'],
                    "text": row['input'],
                    "labels": risk_profiles[i],
                    "processed_at": datetime.now(timezone.utc).isoformat()
                }
                f_out.write(json.dumps(record) + "\n")
            
            # E. Flush & Log
            total_processed += len(valid_rows)
            f_out.flush() # Ensure data is written to disk immediately
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {total_processed} valid rows...", end="\r")

    print(f"\n\nDone! Successfully processed {total_processed} rows.")

if __name__ == "__main__":
    main()
