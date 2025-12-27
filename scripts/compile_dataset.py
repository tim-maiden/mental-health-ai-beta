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
    TEST_FILE
)

# --- DATA FILTERING THRESHOLDS ---
# Safe Prototypes: Only include safe items with density below this threshold
SAFE_DENSITY_THRESHOLD = 0.45 
RISK_DENSITY_THRESHOLD = 0.30

# Hard Negative Upper Bound: Safe items with density above this likely contain label errors or are too ambiguous
HARD_NEGATIVE_UPPER_BOUND = 0.65 

def main():
    print("--- Starting Sequential Dataset Compilation (Decoupled Striding) ---")
    
    # 1. Load Data from Snapshot
    if not os.path.exists(RAW_DATA_FILE):
        print(f"Error: Snapshot file {RAW_DATA_FILE} not found.")
        print("Please run scripts/ingest_data.py first.")
        sys.exit(1)
        
    print(f"Loading data from {RAW_DATA_FILE}...")
    df_all = pd.read_pickle(RAW_DATA_FILE)
    
    # Ensure post_id exists
    if 'post_id' not in df_all.columns:
        print("Warning: 'post_id' column missing. Attempting to synthesize or failing...")
        pass

    # Label Parsing
    df_all['binary_label'] = df_all['dataset_type'].apply(lambda x: 1 if 'mental_health' in str(x) else 0)
    
    # 2. Split Train/Test (BEFORE PCA to prevent leakage)
    print("Splitting by Post ID (preventing data leakage)...")
    if 'post_id' in df_all.columns:
        unique_posts = df_all[['post_id', 'binary_label']].drop_duplicates()
        train_posts, test_posts = train_test_split(unique_posts, test_size=0.15, stratify=unique_posts['binary_label'], random_state=42)
        
        train_df = df_all[df_all['post_id'].isin(train_posts['post_id'])].copy()
        test_df = df_all[df_all['post_id'].isin(test_posts['post_id'])].copy()
    else:
        print("Warning: splitting by row (potential leakage between chunks of same post)...")
        train_df, test_df = train_test_split(df_all, test_size=0.15, stratify=df_all['binary_label'], random_state=42)
    
    print(f"Train Size: {len(train_df)} rows")
    print(f"Test Size:  {len(test_df)} rows")

    # 3. Dimensionality Reduction (PCA) - Fit on Train, Transform Test
    print("Running PCA (Fit on Train, Transform Test)...")
    train_embeddings = np.stack(train_df['embedding_vec'].values)
    test_embeddings = np.stack(test_df['embedding_vec'].values)
    
    train_reduced, pca_model = reduce_dimensions(train_embeddings, n_components=100)
    # Use the fitted pca_model for test set
    test_reduced, _ = reduce_dimensions(test_embeddings, n_components=100, pca_model=pca_model)
    
    train_df['reduced_vec'] = list(train_reduced)
    test_df['reduced_vec'] = list(test_reduced)

    # ==========================================================
    # PHASE 1: PROCESS TRAINING DATA (Margin-Based / Ambiguity Filtering)
    # ==========================================================
    print("\n--- Phase 1: Processing Training Data (Margin-Based Learning) ---")
    
    # A. Build Decoupled Teacher Index (Stride = Window Size)
    WINDOW_SIZE = 3
    print(f"Building Non-Overlapping Teacher Index (Stride={WINDOW_SIZE})...")
    
    if 'chunk_id' in train_df.columns:
        train_df = train_df.sort_values(['post_id', 'chunk_id'])
    
    train_df['post_sequence_idx'] = train_df.groupby('post_id').cumcount()
    teacher_mask = (train_df['post_sequence_idx'] % WINDOW_SIZE) == 0
    
    teacher_df = train_df[teacher_mask].copy()
    print(f"Teacher Index Size: {len(teacher_df)} (Non-overlapping subset)")
    
    teacher_vecs = np.stack(teacher_df['reduced_vec'].values)
    teacher_labels = teacher_df['binary_label'].values
    
    # B. Calc Density (Pass 1)
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
    mask_risk = train_df['binary_label'] == 1
    mask_clean_risk = mask_risk & (train_df['risk_density_p1'] > RISK_DENSITY_THRESHOLD)
    
    train_risk_clean_full = train_df[mask_clean_risk].copy()
    train_safe_all_full = train_df[train_df['binary_label'] == 0].copy()
    
    print(f"Risk Prototypes (Density > {RISK_DENSITY_THRESHOLD}): {mask_risk.sum()} -> {len(train_risk_clean_full)}")
    
    # D. Recalculate Density (Pass 2)
    print("\n--- Pass 2: Identifying Safe Prototypes & Hard Negatives ---")
    
    teacher_risk_clean = train_df[teacher_mask & mask_clean_risk]
    teacher_safe_all = train_df[teacher_mask & (train_df['binary_label'] == 0)]
    
    ref_pass2_df = pd.concat([teacher_risk_clean, teacher_safe_all])
    ref_pass2_vecs = np.stack(ref_pass2_df['reduced_vec'].values)
    ref_pass2_labels = ref_pass2_df['binary_label'].values
    
    print(f"Pass 2 Teacher Index: {len(ref_pass2_df)} (Cleaned Risk + All Safe, Non-Overlapping)")
    
    query_safe_vecs = np.stack(train_safe_all_full['reduced_vec'].values)
    
    density_scores_pass2 = calculate_risk_density(
        query_embeddings=query_safe_vecs,
        reference_embeddings=ref_pass2_vecs,
        reference_labels=ref_pass2_labels,
        k=100
    )
    train_safe_all_full['risk_density_p2'] = density_scores_pass2
    
    # E. Margin-Based Filtering + Hard Negative Mining
    # 1. Safe Prototypes: Low Density
    safe_prototypes = train_safe_all_full[train_safe_all_full['risk_density_p2'] < SAFE_DENSITY_THRESHOLD]
    
    # 2. Hard Negatives: Medium Density (The "Radioactive Zone" we previously dropped)
    # We include these to teach the model to distinguish ambiguous safe content from risk.
    # We cap at UPPER_BOUND to avoid mislabeled data (Safe items that look VERY Risky might actually be Risk).
    safe_hard_negatives = train_safe_all_full[
        (train_safe_all_full['risk_density_p2'] >= SAFE_DENSITY_THRESHOLD) & 
        (train_safe_all_full['risk_density_p2'] < HARD_NEGATIVE_UPPER_BOUND)
    ]
    
    print(f"Safe Prototypes (Density < {SAFE_DENSITY_THRESHOLD}): {len(safe_prototypes)}")
    print(f"Safe Hard Negatives ({SAFE_DENSITY_THRESHOLD} <= Density < {HARD_NEGATIVE_UPPER_BOUND}): {len(safe_hard_negatives)}")
    
    # F. Balance & Merge
    target_risk_size = len(train_risk_clean_full)
    print(f"\n--- Balancing Dataset (Target Class Size: {target_risk_size}) ---")
    
    # We want a mix of Prototypes and Hard Negatives for the Safe Class.
    # e.g., 70% Prototypes, 30% Hard Negatives.
    HARD_NEGATIVE_RATIO = 0.30
    n_hard = int(target_risk_size * HARD_NEGATIVE_RATIO)
    n_proto = target_risk_size - n_hard
    
    # Sample Hard Negatives
    if len(safe_hard_negatives) >= n_hard:
        safe_hard_sampled = safe_hard_negatives.sample(n=n_hard, random_state=42)
    else:
        print(f"Note: Using all available hard negatives ({len(safe_hard_negatives)})")
        safe_hard_sampled = safe_hard_negatives
        n_proto = target_risk_size - len(safe_hard_sampled) # Fill rest with prototypes
        
    # Sample Prototypes
    if len(safe_prototypes) >= n_proto:
        safe_proto_sampled = safe_prototypes.sample(n=n_proto, random_state=42)
    else:
        print(f"Warning: Not enough safe prototypes ({len(safe_prototypes)}) to fill budget ({n_proto}).")
        safe_proto_sampled = safe_prototypes

    safe_combined = pd.concat([safe_proto_sampled, safe_hard_sampled])
    
    # Combine Final
    final_train = pd.concat([
        train_risk_clean_full,
        safe_combined
    ])
    
    # Shuffle
    final_train = final_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final Training Distribution:")
    print(f" - Risk Prototypes: {len(train_risk_clean_full)}")
    print(f" - Safe Combined: {len(safe_combined)} ({len(safe_proto_sampled)} Proto + {len(safe_hard_sampled)} Hard)")
    print(f" - Total: {len(final_train)}")
    
    # ==========================================================
    # PHASE 2: PROCESS TEST DATA
    # ==========================================================
    print("\n--- Phase 2: Processing Test Data ---")
    
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
    
    print(f"Test Set Size: {len(test_df)}")
    
    # ==========================================================
    # PHASE 3: EXPORT
    # ==========================================================
    print("\n--- Exporting Datasets ---")
    
    def save_jsonl(dataframe, filename):
        out = dataframe.rename(columns={'input': 'text'})
        out['label'] = out['binary_label'].astype(int)
        out[['text', 'label']].to_json(filename, orient='records', lines=True)
        print(f"Saved {len(out)} rows to {filename}")

    save_jsonl(final_train, TRAIN_FILE)
    save_jsonl(test_df, TEST_FILE)
    
    print("Done!")

if __name__ == "__main__":
    main()
