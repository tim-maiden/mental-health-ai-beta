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
        df_reddit_chunks = load_reddit_mental_health_dataset()
        embed_and_upload_dataframe_in_batches(
            df_reddit_chunks, 
            'reddit_mental_health_embeddings',
            int_columns=['post_id', 'chunk_id', 'score', 'input_tokens']
        )
        print("=== FINISHED MENTAL HEALTH DATASET PROCESSING ===\n")

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
                print("Continuing... (Duplicate checks are not implemented, so you may get dupes)")

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

