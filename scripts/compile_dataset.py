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

# --- DATA FILTERING THRESHOLDS ---
# Safe Prototypes: Only include safe items with density below this threshold
# Lower threshold = stricter filtering (removes borderline "sad but safe" posts)
# This forces the model to rely more on text signal rather than subreddit context
SAFE_DENSITY_THRESHOLD = 0.15  # Tightened from 0.25 to improve signal sensitivity

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
    # PHASE 1: PROCESS TRAINING DATA (Margin-Based / Ambiguity Filtering)
    # ==========================================================
    print("\n--- Phase 1: Processing Training Data (Margin-Based Learning) ---")
    
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
    
    # C. Filter Risk Prototypes (High Density Risk)
    # Keep Risk items only if risk_density > 0.4 (The Clear Signal)
    
    mask_risk = train_df['binary_label'] == 1
    # Increased threshold to 0.4 for "Prototype" definition
    mask_clean_risk = mask_risk & (train_df['risk_density_p1'] > 0.4)
    
    # Define "Cleaned Risk" subset (from full overlapping)
    train_risk_clean_full = train_df[mask_clean_risk].copy()
    
    # Define "All Safe" subset (from full overlapping) for further filtering
    train_safe_all_full = train_df[train_df['binary_label'] == 0].copy()
    
    print(f"Risk Prototypes (Density > 0.4): {mask_risk.sum()} -> {len(train_risk_clean_full)}")
    
    # D. Recalculate Density (Pass 2)
    # Reference: Non-Overlapping version of (Clean Risk + All Safe)
    # Query: All Safe (to find safe prototypes)
    
    print("\n--- Pass 2: Identifying Safe Prototypes ---")
    
    # Build Pass 2 Teacher (Non-Overlapping)
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
    
    # E. Margin-Based Filtering (The Pivot)
    # Dropping the "Hard Negative Mining" approach.
    # Instead, we select ONLY "Clean Safe" (Low Density).
    # We DROP the "Ambiguous" / "Hard Negative" region entirely from training.
    
    # Safe Prototypes: Density < SAFE_DENSITY_THRESHOLD (0.15)
    # Tightened threshold removes "borderline" safe posts, forcing model to rely on text signal
    safe_prototypes = train_safe_all_full[train_safe_all_full['risk_density_p2'] < SAFE_DENSITY_THRESHOLD]
    
    # Ambiguous / Hard Negatives: Density >= SAFE_DENSITY_THRESHOLD (DROPPED)
    # We log count for info, but do not use them.
    n_dropped_safe = len(train_safe_all_full) - len(safe_prototypes)
    
    print(f"Safe Prototypes (Density < {SAFE_DENSITY_THRESHOLD}): {len(safe_prototypes)}")
    print(f"Ambiguous Safe Dropped (Density >= {SAFE_DENSITY_THRESHOLD}): {n_dropped_safe} (Radioactive Zone)")
    
    # Safety check: Warn if threshold is too aggressive
    if len(safe_prototypes) < 1000:
        print(f"\n⚠️  WARNING: Only {len(safe_prototypes)} safe prototypes found with threshold {SAFE_DENSITY_THRESHOLD}.")
        print("   This may cause class imbalance. Consider relaxing SAFE_DENSITY_THRESHOLD to 0.18-0.20 if training fails.")
    
    # F. Balance & Merge (Margin-Based / No-Fly Zone)
    # ---------------------------------------------------------
    # STRATEGY: Train ONLY on Prototypes. 
    # DROP the "Radioactive Zone" (Safe posts that look like Risk).
    # DROP the "Noise" (Risk posts that don't look like Risk).
    
    # 1. Define Prototypes
    # Risk Prototypes: Already filtered in Step C (train_risk_clean_full)
    # Safe Prototypes: Only Low Density (Pure Safe)
    
    # 2. Balance (Downsample Safe to match Risk)
    target_size = len(train_risk_clean_full)
    print(f"\n--- Balancing Dataset (Target Class Size: {target_size}) ---")
    
    if len(safe_prototypes) > target_size:
        print(f"Downsampling Safe Prototypes ({len(safe_prototypes)} -> {target_size})...")
        # NOTE: Using random sampling to ensure diversity. If you see "jitter" on safe items,
        # check that safe_balanced includes diverse topics (not just technical/bot content).
        safe_balanced = safe_prototypes.sample(n=target_size, random_state=42)
    else:
        # If we don't have enough pure safe posts, we take what we have
        if len(safe_prototypes) < target_size * 0.8:
            print(f"⚠️  WARNING: Safe prototypes ({len(safe_prototypes)}) are significantly fewer than risk prototypes ({target_size}).")
            print("   This will create class imbalance. Consider relaxing SAFE_DENSITY_THRESHOLD.")
        safe_balanced = safe_prototypes
        
    # 3. Combine (CRITICAL: EXCLUDE safe_hard / safe_mid entirely)
    final_train = pd.concat([
        train_risk_clean_full,
        safe_balanced
    ])
    
    # Shuffle
    final_train = final_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final Training Distribution (Margin-Based):")
    print(f" - Risk Prototypes: {len(train_risk_clean_full)}")
    print(f" - Safe Prototypes: {len(safe_balanced)}")
    print(f" - Radioactive/Ambiguous Dropped: {len(train_safe_all_full) - len(safe_prototypes)}")
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
