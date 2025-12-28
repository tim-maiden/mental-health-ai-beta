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
    
    # F. Balance & Merge
    target_risk_size = len(train_risk_clean_full)
    print(f"\n--- Balancing Dataset (Target Class Size: {target_risk_size}) ---")
    
    # We want a mix of Prototypes and Hard Negatives for the Safe Class.
    # e.g., 70% Prototypes, 30% Hard Negatives.
    # [UPDATED] Decreased to 30% to give the model more "easy wins" and boost confidence.
    HARD_NEGATIVE_RATIO = 0.30
    n_hard = int(target_risk_size * HARD_NEGATIVE_RATIO)
    n_proto = target_risk_size - n_hard
    
    # 1. Sample Hard Negatives (Based on Density)
    if len(safe_hard_negatives) >= n_hard:
        safe_hard_sampled = safe_hard_negatives.sample(n=n_hard, random_state=42)
    else:
        print(f"Note: Using all available hard negatives ({len(safe_hard_negatives)})")
        safe_hard_sampled = safe_hard_negatives
        n_proto = target_risk_size - len(safe_hard_sampled) # Fill rest with prototypes
        
    # 2. Sample Prototypes (Emotion-Driven Sampling if available)
    if 'predicted_emotions' in safe_prototypes.columns:
        print("Using Emotion-Driven Sampling for Safe Prototypes...")
        # Explode the list of emotions to sample efficiently
        # Since 'predicted_emotions' is a list, we might want to prioritize "Sadness-like" or "Anxiety-like"
        # but safe content (e.g. "grief" in safe contexts vs risk).
        # Actually, if we want to teach the model distinctions, we want Safe content that MIGHT look risky emotionally.
        # So we upsample "Sadness", "Fear", "Anger" in the Safe class.
        # [UPDATED] The probe now returns simplified sentiments: 'negative', 'positive', 'neutral', 'ambiguous'.
        # We target 'negative' sentiment for hard negatives.
        
        target_emotions = {'negative'}
        
        def has_target_emotion(emotions_list):
            if not isinstance(emotions_list, list): return False
            return any(e in target_emotions for e in emotions_list)
            
        safe_emotional = safe_prototypes[safe_prototypes['predicted_emotions'].apply(has_target_emotion)]
        safe_neutral = safe_prototypes[~safe_prototypes.index.isin(safe_emotional.index)]
        
        print(f"Found {len(safe_emotional)} emotional safe prototypes vs {len(safe_neutral)} neutral/other.")
        
        # We want to oversample the emotional ones to force the model to learn context, not just emotion words.
        # Let's target 50% emotional if possible.
        n_emotional_target = n_proto // 2
        
        if len(safe_emotional) >= n_emotional_target:
            sampled_emotional = safe_emotional.sample(n=n_emotional_target, random_state=42)
        else:
            sampled_emotional = safe_emotional # Take all
            
        n_neutral_target = n_proto - len(sampled_emotional)
        if len(safe_neutral) >= n_neutral_target:
            sampled_neutral = safe_neutral.sample(n=n_neutral_target, random_state=42)
        else:
            sampled_neutral = safe_neutral
            
        safe_proto_sampled = pd.concat([sampled_emotional, sampled_neutral])
        print(f"Sampled {len(sampled_emotional)} emotional + {len(sampled_neutral)} neutral safe prototypes.")
        
    else:
        print("Warning: 'predicted_emotions' column missing. Fallback to random sampling.")
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
    final_train['soft_label'] = compute_soft_labels_knn(final_train, teacher_df, n_neighbors=50, temperature=0.3, subreddit_map=final_subreddit_to_id)
    test_df['soft_label'] = compute_soft_labels_knn(test_df, teacher_df, n_neighbors=50, temperature=0.3, subreddit_map=final_subreddit_to_id)

    # Also save the subreddit mapping
    import json
    mapping_file = os.path.join(DATA_DIR, "subreddit_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump(final_subreddit_to_id, f)
    print(f"Saved subreddit mapping to {mapping_file}")

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
