import os
import sys
import pandas as pd
import numpy as np
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.processing import (
    load_audit_results, 
    create_pools, 
    compile_risk_set, 
    compile_safe_set, 
    create_ambiguous_test_set
)
from src.config import (
    AUDIT_RESULTS_FILE,
    TRAIN_FILE,
    TEST_CLEAN_FILE,
    TEST_AMBIGUOUS_FILE
)
from sklearn.model_selection import train_test_split

INPUT_FILE = AUDIT_RESULTS_FILE
TRAIN_OUTPUT = TRAIN_FILE
TEST_CLEAN_OUTPUT = TEST_CLEAN_FILE
TEST_AMBIGUOUS_OUTPUT = TEST_AMBIGUOUS_FILE
TEST_SIZE = 0.15

def main():
    print(f"--- Starting Hard Negative Mining Dataset Compilation ---")
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = load_audit_results(INPUT_FILE)
    pool_risk, pool_safe = create_pools(df)
    
    print(f"Risk Pool: {len(pool_risk)}")
    print(f"Safe Pool: {len(pool_safe)}")

    print(f"\n--- Splitting Data (Holdout {TEST_SIZE*100}%) ---")
    train_risk, test_risk = train_test_split(pool_risk, test_size=TEST_SIZE, random_state=42)
    train_safe, test_safe = train_test_split(pool_safe, test_size=TEST_SIZE, random_state=42)
    
    print(f"Risk Train: {len(train_risk)}")
    print(f"Safe Train: {len(train_safe)}")

    # 1. Compile Risk
    print("\n--- Compiling Risk Set (Filtering Noisy Labels) ---")
    # Drop Risk items that look completely Safe (purity < 0.1)
    train_risk_final = compile_risk_set(train_risk, min_purity=0.1)
    print(f"Original Risk: {len(train_risk)} -> Filtered Risk: {len(train_risk_final)}")

    # 2. Compile Safe
    print(f"\n--- Compiling Safe Set (Filtering Noisy Labels) ---")
    # Drop Safe items that look completely Risk (purity > 0.9)
    train_safe_filtered = compile_safe_set(train_safe, max_risk_density=0.9)
    
    # 3. Balance Datasets (Optional: Undersample Safe to match Risk ratio? Or just keep all cleaned data?)
    # Usually we want a balanced dataset or slight negative dominance. 
    # Let's keep a 2:1 ratio max, otherwise use all available safe data.
    target_safe_count = len(train_risk_final) * 2
    
    if len(train_safe_filtered) > target_safe_count:
        print(f"Downsampling Safe set from {len(train_safe_filtered)} to {target_safe_count} to maintain 1:2 ratio")
        train_safe_final = train_safe_filtered.sample(n=target_safe_count, random_state=42)
    else:
        train_safe_final = train_safe_filtered

    print(f"Final Safe: {len(train_safe_final)}")

    final_train = pd.concat([train_risk_final, train_safe_final]).sample(frac=1, random_state=42)
    
    # 3. Compile Test Sets
    target_test = min(len(test_risk), len(test_safe))
    test_safe_balanced = test_safe.sample(n=target_test, random_state=42)
    test_risk_balanced = test_risk.sample(n=target_test, random_state=42)
    final_test_clean = pd.concat([test_risk_balanced, test_safe_balanced]).sample(frac=1, random_state=42)

    test_ambiguous = create_ambiguous_test_set(test_risk, test_safe)
    
    print(f"\n--- Ambiguous Test Set Stats ---")
    print(f"Size: {len(test_ambiguous)}")
    
    # Export
    cols = ['text', 'label', 'subreddit']
    def save_jsonl(dataframe, filename):
        out = dataframe.rename(columns={'input': 'text'})
        out[cols].to_json(filename, orient='records', lines=True)
        print(f"Saved {len(out)} rows to {filename}")

    print(f"\n--- Exporting ---")
    save_jsonl(final_train, TRAIN_OUTPUT)
    save_jsonl(final_test_clean, TEST_CLEAN_OUTPUT)
    save_jsonl(test_ambiguous, TEST_AMBIGUOUS_OUTPUT)

if __name__ == "__main__":
    main()
