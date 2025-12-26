import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.metrics import reduce_dimensions, calculate_risk_density
from src.config import (
    RAW_DATA_FILE,
    TRAIN_FILE,
    TEST_CLEAN_FILE,
    TEST_AMBIGUOUS_FILE
)

def main():
    print("--- Starting Sequential Dataset Compilation (Decoupled Striding) ---")
    
    # 1. Load Data from Snapshot
    if not os.path.exists(RAW_DATA_FILE):
        print(f"Error: Snapshot file {RAW_DATA_FILE} not found.")
        print("Please run scripts/ingest_data.py first.")
        sys.exit(1)
        
    print(f"Loading data from {RAW_DATA_FILE}...")
    df_all = pd.read_pickle(RAW_DATA_FILE)
    
    # Ensure post_id and chunk_id exist (Critical for Decoupled Striding)
    if 'post_id' not in df_all.columns:
        print("Warning: 'post_id' column missing. Attempting to synthesize or failing...")
        # Fallback: Create dummy post_id from index if not present, but this defeats the purpose
        # of decoupling. We need real post grouping.
        # If 'subreddit' and some sequential indicator exists, we might guess.
        # But 'ingest_data' should have been updated to fetch it.
        # If we just updated 'storage.py', we need to re-ingest!
        # Assuming the user WILL re-ingest or the file has it.
        pass

    # Label Parsing
    df_all['binary_label'] = df_all['dataset_type'].apply(lambda x: 1 if 'mental_health' in str(x) else 0)
    
    # 2. Global Dimensionality Reduction (PCA)
    print("Running Global PCA...")
    embeddings = np.stack(df_all['embedding_vec'].values)
    reduced_embeddings = reduce_dimensions(embeddings, n_components=100)
    
    df_all['reduced_vec'] = list(reduced_embeddings)
    
    # 3. Split Train/Test
    # Split by POST ID to ensure no chunks from the same post leak into Test
    # This is a cleaner split than random rows.
    if 'post_id' in df_all.columns:
        print("Splitting by Post ID (preventing data leakage)...")
        unique_posts = df_all[['post_id', 'binary_label']].drop_duplicates()
        train_posts, test_posts = train_test_split(unique_posts, test_size=0.15, stratify=unique_posts['binary_label'], random_state=42)
        
        train_df = df_all[df_all['post_id'].isin(train_posts['post_id'])].copy()
        test_df = df_all[df_all['post_id'].isin(test_posts['post_id'])].copy()
    else:
        print("Warning: splitting by row (potential leakage between chunks of same post)...")
        train_df, test_df = train_test_split(df_all, test_size=0.15, stratify=df_all['binary_label'], random_state=42)
    
    print(f"Train Size: {len(train_df)} rows")
    print(f"Test Size:  {len(test_df)} rows")

    # ==========================================================
    # PHASE 1: PROCESS TRAINING DATA (Decoupled Striding)
    # ==========================================================
    print("\n--- Phase 1: Processing Training Data ---")
    
    # A. Build Decoupled Teacher Index (Stride = Window Size)
    # Strategy: Select every 3rd chunk per post (assuming window=3, stride=1 originally)
    # We rely on 'chunk_id' or simply row order per post.
    # Let's use cumcount().
    
    WINDOW_SIZE = 3
    print(f"Building Non-Overlapping Teacher Index (Stride={WINDOW_SIZE})...")
    
    # We sort by post_id and chunk_id to ensure sequence
    if 'chunk_id' in train_df.columns:
        train_df = train_df.sort_values(['post_id', 'chunk_id'])
    
    # Select rows where (sequence_index % WINDOW_SIZE) == 0
    # This ensures we take chunk 0, 3, 6... (Non-overlapping windows)
    train_df['post_sequence_idx'] = train_df.groupby('post_id').cumcount()
    teacher_mask = (train_df['post_sequence_idx'] % WINDOW_SIZE) == 0
    
    teacher_df = train_df[teacher_mask].copy()
    print(f"Teacher Index Size: {len(teacher_df)} (Non-overlapping subset)")
    
    teacher_vecs = np.stack(teacher_df['reduced_vec'].values)
    teacher_labels = teacher_df['binary_label'].values
    
    # B. Calc Density (Pass 1)
    # Query: Full Overlapping Train Set
    # Reference: Non-Overlapping Teacher Set
    print("Calculating Density (Query=Full, Ref=Non-Overlapping)...")
    train_vecs = np.stack(train_df['reduced_vec'].values)
    
    density_scores_pass1 = calculate_risk_density(
        query_embeddings=train_vecs,
        reference_embeddings=teacher_vecs,
        reference_labels=teacher_labels,
        k=100
    )
    train_df['risk_density_p1'] = density_scores_pass1
    
    # C. Filter Risk (Cleaning the Teacher)
    # Keep Risk items only if risk_density > 0.25 (Must look somewhat risky)
    # Filter the TEACHER set first? Or the Full set?
    # Guidance: "Filter Risk (Clean the Teacher)"
    # We want to identify valid risk items to use as the reference for the NEXT pass.
    # So we filter the *entire* train set to find valid risk, but when building the index for Pass 2,
    # we should arguably still maintain non-overlapping property if possible, OR just use the cleaned set
    # as the ground truth.
    # Guidance says: "Build a new Non-Overlapping Index using only the risk_cleaned set + all_safe set."
    
    # Let's identify "Clean Risk" from the full set
    mask_risk = train_df['binary_label'] == 1
    mask_clean_risk = mask_risk & (train_df['risk_density_p1'] > 0.25)
    
    # Define "Cleaned Risk" subset (from full overlapping)
    train_risk_clean_full = train_df[mask_clean_risk].copy()
    
    # Define "All Safe" subset (from full overlapping)
    train_safe_all_full = train_df[train_df['binary_label'] == 0].copy()
    
    print(f"Filtered Risk Set: {mask_risk.sum()} -> {len(train_risk_clean_full)} (Removed noisy samples)")
    
    # D. Recalculate Density (Pass 2)
    # Reference: Non-Overlapping version of (Clean Risk + All Safe)
    # Query: All Safe (to find hard negatives)
    
    print("\n--- Pass 2: Hard Negative Mining ---")
    
    # Build Pass 2 Teacher (Non-Overlapping)
    # We need to subsample the Cleaned Risk and All Safe again to ensure non-overlap
    # We can reuse the 'teacher_mask' logic
    
    # Get the subsets of the ORIGINAL teacher_df that survived the filter (for Risk) 
    # and all Safe teacher items.
    # This ensures the reference index is still non-overlapping.
    
    # Risk Teacher (Cleaned): Intersection of Teacher Mask AND Clean Risk Mask
    teacher_risk_clean = train_df[teacher_mask & mask_clean_risk]
    
    # Safe Teacher (All): Intersection of Teacher Mask AND Safe Mask
    teacher_safe_all = train_df[teacher_mask & (train_df['binary_label'] == 0)]
    
    ref_pass2_df = pd.concat([teacher_risk_clean, teacher_safe_all])
    ref_pass2_vecs = np.stack(ref_pass2_df['reduced_vec'].values)
    ref_pass2_labels = ref_pass2_df['binary_label'].values
    
    print(f"Pass 2 Teacher Index: {len(ref_pass2_df)} (Cleaned Risk + All Safe, Non-Overlapping)")
    
    # Query: Full Overlapping Safe Set
    query_safe_vecs = np.stack(train_safe_all_full['reduced_vec'].values)
    
    density_scores_pass2 = calculate_risk_density(
        query_embeddings=query_safe_vecs,
        reference_embeddings=ref_pass2_vecs,
        reference_labels=ref_pass2_labels,
        k=100
    )
    train_safe_all_full['risk_density_p2'] = density_scores_pass2
    
    # E. Mining
    safe_hard = train_safe_all_full[train_safe_all_full['risk_density_p2'] > 0.5]
    safe_easy = train_safe_all_full[train_safe_all_full['risk_density_p2'] < 0.2]
    safe_mid  = train_safe_all_full[(train_safe_all_full['risk_density_p2'] >= 0.2) & (train_safe_all_full['risk_density_p2'] <= 0.5)]
    
    print(f"Safe Hard Negatives (>0.5): {len(safe_hard)}")
    print(f"Safe Easy Negatives (<0.2): {len(safe_easy)}")
    
    # F. Balance & Merge (Aggressive Hard Negative Oversampling)
    # Target: 33% Risk, 33% Safe Hard, 33% Safe Easy
    
    target_size = len(train_risk_clean_full)
    print(f"\n--- Balancing Dataset (Target Class Size: {target_size}) ---")
    
    # 1. Oversample Hard Negatives (The Critical Fix)
    n_hard = len(safe_hard)
    if n_hard > 0:
        print(f"Oversampling Hard Negatives ({n_hard} -> {target_size})...")
        # Use replace=True to duplicate samples until we reach target size
        safe_hard_upsampled = safe_hard.sample(n=target_size, replace=True, random_state=42)
    else:
        print("Warning: No Hard Negatives found! Using Empty DataFrame.")
        safe_hard_upsampled = safe_hard
        
    # 2. Downsample Easy Negatives (Maintenance)
    pool_easy_mid = pd.concat([safe_easy, safe_mid])
    if len(pool_easy_mid) > target_size:
        print(f"Downsampling Safe Easy/Mid ({len(pool_easy_mid)} -> {target_size})...")
        safe_easy_balanced = pool_easy_mid.sample(n=target_size, random_state=42)
    else:
        safe_easy_balanced = pool_easy_mid

    # 3. Combine
    final_train = pd.concat([
        train_risk_clean_full,
        safe_hard_upsampled, 
        safe_easy_balanced
    ])
    
    # Shuffle
    final_train = final_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final Training Distribution:")
    print(f" - Risk: {len(train_risk_clean_full)}")
    print(f" - Safe (Hard): {len(safe_hard_upsampled)}")
    print(f" - Safe (Easy/Mid): {len(safe_easy_balanced)}")
    print(f" - Total: {len(final_train)}")
    
    # ==========================================================
    # PHASE 2: PROCESS TEST DATA
    # ==========================================================
    print("\n--- Phase 2: Processing Test Data ---")
    
    # For Test Eval, we can use the Final Train set as the teacher (it's what the model learned from)
    # Or a non-overlapping version of it?
    # Usually standard KNN against the training set is fine here.
    
    teacher_vecs = np.stack(final_train['reduced_vec'].values)
    teacher_labels = final_train['binary_label'].values
    test_vecs = np.stack(test_df['reduced_vec'].values)
    
    test_density = calculate_risk_density(
        query_embeddings=test_vecs,
        reference_embeddings=teacher_vecs,
        reference_labels=teacher_labels,
        k=100
    )
    test_df['risk_density'] = test_density
    
    mask_test_risk_clean = (test_df['binary_label'] == 1) & (test_df['risk_density'] > 0.5)
    mask_test_safe_clean = (test_df['binary_label'] == 0) & (test_df['risk_density'] < 0.2)
    
    test_clean = pd.concat([
        test_df[mask_test_risk_clean],
        test_df[mask_test_safe_clean]
    ]).sample(frac=1, random_state=42)
    
    test_ambiguous = test_df.drop(test_clean.index).sample(frac=1, random_state=42)
    
    print(f"Test Clean: {len(test_clean)}")
    print(f"Test Ambiguous: {len(test_ambiguous)}")
    
    # ==========================================================
    # PHASE 3: EXPORT
    # ==========================================================
    print("\n--- Exporting Datasets ---")
    cols = ['text', 'label', 'subreddit'] 
    
    def save_jsonl(dataframe, filename):
        out = dataframe.rename(columns={'input': 'text'})
        out['label'] = out['binary_label'].astype(int)
        out[['text', 'label']].to_json(filename, orient='records', lines=True)
        print(f"Saved {len(out)} rows to {filename}")

    save_jsonl(final_train, TRAIN_FILE)
    save_jsonl(test_clean, TEST_CLEAN_FILE)
    save_jsonl(test_ambiguous, TEST_AMBIGUOUS_FILE)
    
    print("Done!")

if __name__ == "__main__":
    main()
