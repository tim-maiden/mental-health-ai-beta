
import sys
import os
import pandas as pd
from itertools import islice

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.data.loaders import yield_reddit_mental_health_dataset

def run_bounded_test(limit_chunks=2000):
    print(f"Running bounded test (limit={limit_chunks} chunks)...")
    
    generator = yield_reddit_mental_health_dataset(batch_size=500)
    
    total_processed = 0
    collected_rows = []
    
    for df_batch in generator:
        print(f"Received batch of size {len(df_batch)}")
        
        # Verify columns exist
        required = ['author', 'created_utc', 'score', 'input', 'subreddit', 'title', 'label']
        missing = [c for c in required if c not in df_batch.columns]
        if missing:
            print(f"ERROR: Batch missing columns: {missing}")
            break
            
        collected_rows.append(df_batch)
        total_processed += len(df_batch)
        
        # Check Label distribution in this batch
        labels = df_batch['label'].unique()
        print(f"  Labels in batch: {labels}")
        
        if total_processed >= limit_chunks:
            print(f"Reached limit ({total_processed} >= {limit_chunks}). Stopping.")
            break
            
    if collected_rows:
        final_df = pd.concat(collected_rows)
        print("\nFinal Data Sample (First 5 rows):")
        print(final_df[['subreddit', 'label', 'author', 'title']].head().to_string())
        print(f"\nTotal rows processed: {len(final_df)}")
        
        # Check if we got any "unlabeled" vs "actual labels"
        print("\nOverall Label Distribution in Sample:")
        print(final_df['label'].value_counts())

if __name__ == "__main__":
    run_bounded_test()

