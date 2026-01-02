import argparse
import sys
import os

# Add the project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loaders import (
    load_lmsys_chat_dataset,
    load_wildchat_dataset,
    load_reddit_mental_health_dataset,
    load_reddit_control_dataset,
    load_goemotions_dataset
)
from src.data.storage import embed_and_upload_dataframe_in_batches

def main():
    """Main function to run the data processing and upload pipeline with CLI flags."""
    parser = argparse.ArgumentParser(description="Embed and upload datasets to Supabase.")
    parser.add_argument("--lmsys", action="store_true", help="Process and upload LMSYS Chat dataset")
    parser.add_argument("--wildchat", action="store_true", help="Process and upload WildChat dataset")
    parser.add_argument("--mental-health", action="store_true", help="Process and upload Reddit Mental Health dataset")
    parser.add_argument("--controls", action="store_true", help="Process and upload Reddit Safe Control dataset")
    parser.add_argument("--goemotions", action="store_true", help="Process and upload GoEmotions dataset")
    parser.add_argument("--reset", action="store_true", help="Clear the target table before uploading (WARNING: Destructive)")
    
    args = parser.parse_args()

    # If no flags are provided, print help
    if not any([args.lmsys, args.wildchat, args.mental_health, args.controls, args.goemotions]):
        print("No dataset selected. Please use at least one flag:")
        print("  --lmsys          : Run LMSYS Chat dataset")
        print("  --wildchat       : Run WildChat dataset")
        print("  --mental-health  : Run Reddit Mental Health dataset")
        print("  --controls       : Run Reddit Safe Control dataset")
        print("  --goemotions     : Run GoEmotions dataset")
        return

    # 1. Process lmsys chat data
    if args.lmsys:
        print("\n=== STARTING LMSYS DATASET PROCESSING ===")
        df_chat_turns = load_lmsys_chat_dataset()
        embed_and_upload_dataframe_in_batches(
            df_chat_turns, 
            'lmsys_chat_embeddings', 
            int_columns=['turn_id', 'input_tokens']
        )
        print("=== FINISHED LMSYS DATASET PROCESSING ===\n")

    # 2. Process WildChat data
    if args.wildchat:
        print("\n=== STARTING WILDCHAT DATASET PROCESSING ===")
        # Limit to 100,000 conversations to target ~1M chunks
        limit = 100000
        df_chat_chunks = load_wildchat_dataset(limit=limit)
        embed_and_upload_dataframe_in_batches(
            df_chat_chunks, 
            'wildchat_embeddings', 
            int_columns=['turn_id', 'chunk_id', 'input_tokens']
        )
        print("=== FINISHED WILDCHAT DATASET PROCESSING ===\n")

    # 3. Process Reddit mental health data
    if args.mental_health:
        print("\n=== STARTING MENTAL HEALTH DATASET PROCESSING ===")
        # Use the generator to process in batches
        from src.data.loaders import yield_reddit_mental_health_dataset
        from src.config import PROGRESS_FILE
        import json
        
        # Load progress manually to handle generator skipping
        processed_rows = 0
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
                processed_rows = progress.get('reddit_mental_health_embeddings', 0)
        
        print(f"Resuming from global row count: {processed_rows}")
        
        batch_counter = 0
        total_chunks = 0
        current_offset = 0
        
        # Adjust batch size for memory safety (e.g. 5000 chunks)
        # Sampling to reduce dataset size.
        # Limit at 1.25M to ensure full alphabetical coverage
        generator = yield_reddit_mental_health_dataset(
            batch_size=5000,
            sample_rate=0.25,
            limit=1250000
        )
        
        for df_batch in generator:
            batch_len = len(df_batch)
            
            # Check if this entire batch has already been processed
            if current_offset + batch_len <= processed_rows:
                print(f"Skipping batch (Global range {current_offset}-{current_offset+batch_len} already processed)")
                current_offset += batch_len
                continue
                
            # Handle partial overlap.
            if current_offset < processed_rows:
                skip_in_batch = processed_rows - current_offset
                print(f"Skipping first {skip_in_batch} rows of current batch...")
                df_batch = df_batch.iloc[skip_in_batch:]
                current_offset += skip_in_batch # Now current_offset == processed_rows
            
            batch_counter += 1
            print(f"\n--- Processing Global Batch {batch_counter} (Global Offset: {current_offset}) ---")
            
            # Directly use embed_dataframe and supabase insert to avoid progress tracking conflicts in the batch loader.
            
            from src.data.storage import embed_dataframe, save_progress, load_progress, supabase
            import numpy as np
            import pandas as pd
            
            # 1. Embed
            df_batch = embed_dataframe(df_batch, desc=f"Embedding batch {batch_counter}")
            
            # 2. Upload
            # Prepare data
            df_batch = df_batch.replace({np.nan: None, pd.NA: None})
            
            # Ensure int columns
            int_columns = ['chunk_id', 'score', 'input_tokens']
            for col in int_columns:
                if col in df_batch.columns:
                    df_batch.loc[:, col] = pd.to_numeric(df_batch[col], errors='coerce').astype('Int64')
            
            records = df_batch.to_dict(orient='records')
            records = [r for r in records if r.get('embedding') is not None]
            
            if records:
                try:
                    # Upload in smaller chunks to Supabase to avoid timeout
                    # Use globally configured BATCH_SIZE (250)
                    from src.config import BATCH_SIZE as SUB_BATCH_SIZE
                    import time
                    
                    for i in range(0, len(records), SUB_BATCH_SIZE):
                        sub_batch = records[i:i+SUB_BATCH_SIZE]
                        
                        # Retry logic for Supabase inserts
                        max_retries = 5
                        for attempt in range(max_retries):
                            try:
                                supabase.table('reddit_mental_health_embeddings').insert(sub_batch).execute()
                                print(f"Uploaded {len(sub_batch)} rows...", end="\r")
                                break # Success, exit retry loop
                            except Exception as e:
                                print(f"\nError uploading sub-batch (Attempt {attempt+1}/{max_retries}): {e}")
                                if attempt < max_retries - 1:
                                    sleep_time = 5 * (2 ** attempt) # Exponential backoff: 5, 10, 20, 40s
                                    print(f"Retrying in {sleep_time} seconds...")
                                    time.sleep(sleep_time)
                                else:
                                    print("Max retries reached. Failing batch.")
                                    raise e # Re-raise to trigger outer exception handler
                                    
                    print("\nBatch upload complete.")
                    
                    # Update Progress
                    current_offset += len(df_batch) # df_batch here is the *remaining* part we processed
                    
                    # Reload progress to be safe (though we are the only writer)
                    p = load_progress()
                    p['reddit_mental_health_embeddings'] = current_offset
                    save_progress(p)
                    
                except Exception as e:
                    print(f"Error uploading batch: {e}")
                    # Handle batch errors gracefully to preserve partial progress.
                    break
            
            total_chunks += len(df_batch)
            
        print(f"=== FINISHED MENTAL HEALTH DATASET PROCESSING (Total Chunks: {total_chunks}) ===\n")

    # 3. Process Reddit Control data
    if args.controls:
        print("\n=== STARTING REDDIT CONTROL DATASET PROCESSING ===")
        
        # Handle Reset
        if args.reset:
            print("WARNING: --reset flag detected. Clearing 'reddit_safe_embeddings' table...")
            from src.core.clients import supabase
            # Delete all rows (id is usually > 0)
            try:
                # neq -1 should match all valid IDs
                supabase.table('reddit_safe_embeddings').delete().neq('id', -1).execute()
                print("Table cleared.")
                
                # Also reset progress.json for this table
                import json
                from src.config import PROGRESS_FILE
                if os.path.exists(PROGRESS_FILE):
                    with open(PROGRESS_FILE, 'r') as f:
                        progress = json.load(f)
                    if 'reddit_safe_embeddings' in progress:
                        del progress['reddit_safe_embeddings']
                        with open(PROGRESS_FILE, 'w') as f:
                            json.dump(progress, f)
                    print("Progress reset.")
            except Exception as e:
                print(f"Error clearing table: {e}")
                print("Note: Duplicate checks are not implemented.")

        df_control = load_reddit_control_dataset()
        embed_and_upload_dataframe_in_batches(
            df_control,
            'reddit_safe_embeddings',
            int_columns=['input_tokens', 'chunk_id']
        )
        print("=== FINISHED REDDIT CONTROL DATASET PROCESSING ===\n")

    # 4. Process GoEmotions data
    if args.goemotions:
        print("\n=== STARTING GOEMOTIONS DATASET PROCESSING ===")
        df_goemotions = load_goemotions_dataset()
        embed_and_upload_dataframe_in_batches(
            df_goemotions,
            'goemotions_embeddings',
            int_columns=['chunk_id', 'input_tokens']
        )
        print("=== FINISHED GOEMOTIONS DATASET PROCESSING ===\n")

    print("Script finished.")

if __name__ == "__main__":
    main()

