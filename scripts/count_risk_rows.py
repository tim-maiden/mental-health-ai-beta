import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loaders import yield_reddit_mental_health_dataset

def main():
    print("Counting risk rows (chunks and unique posts) based on parameters in src/data/loaders.py...")
    
    total_chunks = 0
    unique_posts = set()
    
    # We use a large batch size for faster iteration, though the generator defaults to 1000
    # sample_rate=1.0 ensures we count everything matching the filters
    # User's recent request for clarification:
    # 3.8M chunks total. 25% of that should be ~950k.
    # The upload script is likely seeing >1M because of how sample_rate interacts with the generator logic
    # or because the user is seeing the *global* loop index and confusing it with actual yield.
    # Let's count explicitly with 0.25 to verify.
    
    print("\n--- Counting with sample_rate=1.0 (Total Population) ---")
    generator_full = yield_reddit_mental_health_dataset(batch_size=5000, sample_rate=1.0, limit=None)
    
    total_chunks_full = 0
    unique_posts_full = set()
    
    for i, batch in enumerate(generator_full):
        batch_len = len(batch)
        total_chunks_full += batch_len
        if 'post_id' in batch.columns:
            unique_posts_full.update(batch['post_id'].unique())
        print(f"Full Batch {i+1}: {batch_len} chunks. Total: {total_chunks_full}", end='\r')
    
    print(f"\nTotal Risk Chunks (100%): {total_chunks_full}")
    print(f"Total Unique Posts (100%): {len(unique_posts_full)}")

    print("\n\n--- Counting with sample_rate=0.25 (Simulation) ---")
    generator_sample = yield_reddit_mental_health_dataset(batch_size=5000, sample_rate=0.25, limit=None)
    
    total_chunks_sample = 0
    unique_posts_sample = set()
    
    for i, batch in enumerate(generator_sample):
        batch_len = len(batch)
        total_chunks_sample += batch_len
        if 'post_id' in batch.columns:
            unique_posts_sample.update(batch['post_id'].unique())
        print(f"Sample Batch {i+1}: {batch_len} chunks. Total: {total_chunks_sample}", end='\r')

    print(f"\nTotal Risk Chunks (25%): {total_chunks_sample}")
    print(f"Total Unique Posts (25%): {len(unique_posts_sample)}")


if __name__ == "__main__":
    main()

