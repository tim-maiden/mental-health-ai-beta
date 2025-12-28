import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.metrics import reduce_dimensions, calculate_risk_density
from src.config import (
    RAW_DATA_FILE,
    TRAIN_FILE,
    TEST_FILE,
    DATA_DIR
)

# --- DATA FILTERING THRESHOLDS ---
# Local Purity Strategy:
# k=20 (was 100): Allows small, subtle clusters of unique risk to survive.
# RISK > 0.60 (was 0.30): Strict uniqueness. Only keeps text that is MAJORITY risk in its local neighborhood.
NEIGHBOR_K = 20 
SAFE_DENSITY_THRESHOLD = 0.45 
RISK_DENSITY_THRESHOLD = 0.60

# Hard Negative Upper Bound: Safe items with density above this likely contain label errors or are too ambiguous
# UPDATED: Raised to 0.98 to force "Extreme Hard Negatives" (like Hiking vs Anxiety) into the dataset.
# We trust our Ground Truth labels more than the embedding similarity.
HARD_NEGATIVE_UPPER_BOUND = 0.98 

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

    # 3. Dimensionality Reduction (PCA) - SKIPPED for Higher Accuracy
    # We now use full embeddings for both density calculation and soft label generation
    # to prevent semantic loss (e.g. Hiking vs Anxiety).
    print("Skipping PCA (Using Full Embeddings)...")
    
    # We won't create 'reduced_vec', we'll just use 'embedding_vec' directly in downstream steps.
    # To minimize refactoring, we can alias it if needed, but better to update calls.

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
    
    teacher_vecs = np.stack(teacher_df['embedding_vec'].values)
    teacher_labels = teacher_df['binary_label'].values
    
    # B. Calc Density (Pass 1)
    print(f"Calculating Density (Query=Full, Ref=Non-Overlapping, k={NEIGHBOR_K})...")
    train_vecs = np.stack(train_df['embedding_vec'].values)
    
    density_scores_pass1 = calculate_risk_density(
        query_embeddings=train_vecs,
        reference_embeddings=teacher_vecs,
        reference_labels=teacher_labels,
        k=NEIGHBOR_K
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
    ref_pass2_vecs = np.stack(ref_pass2_df['embedding_vec'].values)
    ref_pass2_labels = ref_pass2_df['binary_label'].values
    
    print(f"Pass 2 Teacher Index: {len(ref_pass2_df)} (Cleaned Risk + All Safe, Non-Overlapping)")
    
    query_safe_vecs = np.stack(train_safe_all_full['embedding_vec'].values)
    
    density_scores_pass2 = calculate_risk_density(
        query_embeddings=query_safe_vecs,
        reference_embeddings=ref_pass2_vecs,
        reference_labels=ref_pass2_labels,
        k=NEIGHBOR_K
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
    
    # F. Ratio-Based Balancing (1 Risk : 5 Safe)
    print(f"\n--- Compiling Dataset with Enforced 1:5 Ratio ---")
    
    # 1. Determine Target Safe Count
    risk_count = len(train_risk_clean_full)
    target_safe_count = risk_count * 5
    print(f"Target Safe Count: {target_safe_count} (Risk Count: {risk_count})")
    
    # 2. Prioritize Hard Negatives (Keep ALL of them for boundary definition)
    # These are the most valuable samples.
    safe_hard_sampled = safe_hard_negatives
    
    # 3. Fill Remainder with Prototypes
    needed_prototypes = target_safe_count - len(safe_hard_sampled)
    
    if needed_prototypes > 0:
        # Sample from the prototypes to fill the quota
        # If we had emotion labels, we would prioritize "Joy/Optimism" here.
        # Since we don't (or choose not to use them), random sampling is the next best thing.
        if len(safe_prototypes) > needed_prototypes:
            safe_proto_sampled = safe_prototypes.sample(n=needed_prototypes, random_state=42)
        else:
            safe_proto_sampled = safe_prototypes
    else:
        # Rare case where we have huge number of hard negatives
        safe_proto_sampled = pd.DataFrame() 

    safe_combined = pd.concat([safe_proto_sampled, safe_hard_sampled])
    
    # Combine Final
    final_train = pd.concat([
        train_risk_clean_full,
        safe_combined
    ])
    
    # Shuffle
    final_train = final_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final Training Distribution (RATIO ENFORCED):")
    print(f" - Risk Prototypes: {len(train_risk_clean_full)}")
    print(f" - Safe Combined: {len(safe_combined)}")
    print(f" - Ratio: 1 Risk : {len(safe_combined)/len(train_risk_clean_full):.1f} Safe")
    
    # ==========================================================
    # PHASE 1.5: GENERATE SOFT LABELS (EMBEDDING DISTILLATION)
    # ==========================================================
    print("\n--- Phase 1.5: Generating Soft Labels (Weighted k-NN) ---")
    
    if 'subreddit' not in teacher_df.columns:
        print("Error: 'subreddit' column missing. Cannot generate soft labels.")
        sys.exit(1)

    # Prepare Subreddit Map
    unique_subreddits = sorted(train_df['subreddit'].unique())
    final_subreddit_to_id = {sub: i for i, sub in enumerate(unique_subreddits)}
    print(f"Found {len(unique_subreddits)} unique subreddits.")
    
    def compute_soft_labels_knn(query_df, teacher_df, n_neighbors=50, temperature=0.3, subreddit_map=None):
        """
        Computes soft labels using Weighted k-NN.
        Returns: List of probability distributions (size: N_samples x N_classes).
        """
        print(f"Computing k-NN Soft Labels (k={n_neighbors}, temp={temperature})...")
        
        # 1. Prepare Vectors
        query_vecs = np.stack(query_df['embedding_vec'].values)
        teacher_vecs = np.stack(teacher_df['embedding_vec'].values)
        teacher_subs = teacher_df['subreddit'].map(subreddit_map).values  # Map to Int IDs
        num_classes = len(subreddit_map)

        # 2. Build Index & Query
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=-1)
        nbrs.fit(teacher_vecs)
        distances, indices = nbrs.kneighbors(query_vecs)
        
        # 3. Compute Soft Labels
        soft_labels = []
        
        for i in range(len(query_vecs)):
            dists = distances[i]
            neighbor_indices = indices[i]
            neighbor_classes = teacher_subs[neighbor_indices]
            
            # Inverse Distance Weighting (Add epsilon)
            weights = 1.0 / (dists + 1e-6)
            
            # Aggregate weights by class
            class_scores = np.zeros(num_classes)
            np.add.at(class_scores, neighbor_classes, weights)
            
            # Apply Temperature Scaling to the aggregated scores (logits)
            logits = class_scores / temperature
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits)) # Stability
            probs = exp_logits / np.sum(exp_logits)
            
            soft_labels.append(probs.tolist())
            
        return soft_labels

    # Apply to Final Train and Test using the teacher set
    # Using T=1.0 as advised to preserve dark knowledge (relationships between classes)
    final_train['soft_label'] = compute_soft_labels_knn(final_train, teacher_df, n_neighbors=50, temperature=1.0, subreddit_map=final_subreddit_to_id)
    test_df['soft_label'] = compute_soft_labels_knn(test_df, teacher_df, n_neighbors=50, temperature=1.0, subreddit_map=final_subreddit_to_id)

    # Also save the subreddit mapping
    import json
    mapping_file = os.path.join(DATA_DIR, "subreddit_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump(final_subreddit_to_id, f)
    print(f"Saved subreddit mapping to {mapping_file}")

    # Save Risk Indices for Hierarchical Loss
    # Identify risk subreddits from the risk dataframe (using the clean filtered set)
    risk_subs = set(train_risk_clean_full['subreddit'].unique())
    risk_indices = [idx for sub, idx in final_subreddit_to_id.items() if sub in risk_subs]
    
    risk_indices_file = os.path.join(DATA_DIR, "risk_indices.json")
    with open(risk_indices_file, "w") as f:
        json.dump(risk_indices, f)
    print(f"Saved {len(risk_indices)} risk indices to {risk_indices_file}")

    # ==========================================================
    # PHASE 2: PROCESS TEST DATA
    # ==========================================================
    print("\n--- Phase 2: Processing Test Data ---")
    
    teacher_vecs = np.stack(final_train['embedding_vec'].values)
    teacher_labels = final_train['binary_label'].values
    test_vecs = np.stack(test_df['embedding_vec'].values)
    
    test_density = calculate_risk_density(
        query_embeddings=test_vecs,
        reference_embeddings=teacher_vecs,
        reference_labels=teacher_labels,
        k=NEIGHBOR_K
    )
    test_df['risk_density'] = test_density
    
    print(f"Test Set Size: {len(test_df)}")
    
    # ==========================================================
    # PHASE 3: EXPORT
    # ==========================================================
    print("\n--- Exporting Datasets ---")
    
    def save_jsonl(dataframe, filename):
        out = dataframe.rename(columns={'input': 'text'})
        # Save soft labels as the primary label for training
        # We also keep the hard 'binary_label' for evaluation metrics if needed, but the model trainer
        # will look for 'label' or 'labels'. 
        # We will name the soft label column 'label' so the HuggingFace trainer picks it up automatically.
        # BUT: HF Trainer expects 'label' to be int for classification or float for regression.
        # For Multi-Label/Soft-Target, we usually pass a float tensor.
        
        # Let's keep 'label' as the binary int for backward compatibility/metrics
        # and 'soft_label' as the distribution.
        # We will modify the Training script to look for 'soft_label'.
        
        out['label'] = out['binary_label'].astype(int)
        
        # Ensure soft_label is included
        cols_to_save = ['text', 'label', 'soft_label']
        if 'subreddit' in out.columns:
            cols_to_save.append('subreddit')
            
        out[cols_to_save].to_json(filename, orient='records', lines=True)
        print(f"Saved {len(out)} rows to {filename}")

    save_jsonl(final_train, TRAIN_FILE)
    save_jsonl(test_df, TEST_FILE)
    
    print("Done!")

if __name__ == "__main__":
    main()
