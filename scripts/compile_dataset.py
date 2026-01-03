import os
import sys
import pandas as pd
import numpy as np
import json
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: faiss not installed. Please install faiss-cpu.")
    FAISS_AVAILABLE = False

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    RAW_DATA_FILE,
    TRAIN_FILE,
    TEST_FILE,
    DATA_DIR,
    NEIGHBOR_K,
    RISK_DENSITY_THRESHOLD,
    SAFE_DENSITY_THRESHOLD,
    SEED,
    HIGH_EMOTION_THRESHOLD,
    LOW_EMOTION_THRESHOLD
)

def get_faiss_index(d, ref_vecs):
    """
    Creates and returns a FAISS index. 
    FORCED CPU MODE (HNSW) to avoid H100/CUDA kernel compatibility issues.
    """
    print("  > Using CPU Index (IndexHNSWFlat) for compatibility...")
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.train(ref_vecs)
    index.add(ref_vecs)
    return index

def calculate_density_faiss(query_vecs, ref_vecs, k=NEIGHBOR_K):
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not installed.")

    d = ref_vecs.shape[1]
    query_vecs = query_vecs.astype(np.float32)
    ref_vecs = ref_vecs.astype(np.float32)
    
    faiss.normalize_L2(query_vecs)
    faiss.normalize_L2(ref_vecs) 
    
    index = get_faiss_index(d, ref_vecs)
    print(f"  > Searching Index (query_size={len(query_vecs)}, k={k})...")
    D, I = index.search(query_vecs, k)
    
    densities = np.mean(D, axis=1)
    return densities

def compute_soft_labels_faiss(query_vecs, teacher_vecs, teacher_df, n_neighbors=50, temperature=1.0, subreddit_map=None):
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not installed.")

    print(f"Computing k-NN Soft Labels (k={n_neighbors}, temp={temperature})...")
    
    query_vecs = query_vecs.astype(np.float32)
    teacher_vecs = teacher_vecs.astype(np.float32)
    
    faiss.normalize_L2(query_vecs)
    faiss.normalize_L2(teacher_vecs)
    
    teacher_subs = teacher_df['subreddit'].map(subreddit_map).values.astype(int)
    num_classes = len(subreddit_map)
    d = teacher_vecs.shape[1]

    index = get_faiss_index(d, teacher_vecs)
    distances, indices = index.search(query_vecs, n_neighbors)
    
    soft_labels = []
    
    similarities = distances 
    weights = np.exp(similarities / temperature) 
    
    for i in range(len(query_vecs)):
        neighbor_indices = indices[i]
        neighbor_weights = weights[i]
        
        label_counts = np.zeros(num_classes)
        for idx, w in zip(neighbor_indices, neighbor_weights):
            if idx != -1:
                sub_id = teacher_subs[idx]
                label_counts[sub_id] += w
                
        if np.sum(label_counts) > 0:
            label_dist = label_counts / np.sum(label_counts)
        else:
            label_dist = np.ones(num_classes) / num_classes
            
        soft_labels.append(label_dist.tolist())
        
    return np.array(soft_labels)

def get_emotion_score(row, target):
    scores = row.get('emotion_scores', {})
    if isinstance(scores, str):
        try:
            scores = json.loads(scores)
        except:
            scores = {}
    
    if isinstance(scores, dict):
        return float(scores.get(target, 0.0))
        
    ems = row.get('predicted_emotions', [])
    if isinstance(ems, list) and target in ems:
        return 1.0
    return 0.0

def main():
    print("--- Starting Dataset Compilation (Memory Optimized) ---")
    
    if not FAISS_AVAILABLE:
        print("Error: FAISS is required.")
        sys.exit(1)
        
    print(f"Loading data from Hugging Face Hub (tim-maiden/mental-health-ai)...")
    try:
        # Load dataset
        dataset = load_dataset("tim-maiden/mental-health-ai", split="train")
        print(f"Loaded Dataset: {len(dataset)} rows.")

        # Ensure output directory exists
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            print(f"Verified data directory: {DATA_DIR}")
        except Exception as e:
            print(f"Error creating data directory {DATA_DIR}: {e}")
            sys.exit(1)

        # A. Handle Embeddings (Robust Loading)
        print("Mapping embeddings to Numpy (Zero-Copy mode)...")
        dataset.set_format(type='numpy', columns=['embedding'])
        
        all_embeddings = dataset['embedding'] 
        
        # --- FIX START: Handle PyArrow/List Types Explicitly ---
        if hasattr(all_embeddings, "to_numpy"):
            # Handle PyArrow Columns (common in newer datasets versions)
            print("  > Converting PyArrow Column to Numpy...")
            all_embeddings = all_embeddings.to_numpy()
        
        if not isinstance(all_embeddings, np.ndarray):
            print(f"  > Warning: Embeddings are {type(all_embeddings)}, forcing numpy conversion...")
            # Fallback for lists (slower but safe)
            all_embeddings = np.array(all_embeddings)
        
        # Now safe to check dtype
        if all_embeddings.dtype != np.float32:
            print("  > Casting embeddings to float32...")
            all_embeddings = all_embeddings.astype(np.float32)
        # --- FIX END ---
            
        print(f"Embedding Matrix Shape: {all_embeddings.shape} ({all_embeddings.nbytes / 1e9:.2f} GB)")

        # B. Handle Metadata (Pandas - WITHOUT Embeddings)
        print("Loading metadata to Pandas...")
        dataset_meta = dataset.remove_columns(['embedding'])
        dataset_meta.reset_format() 
        df_all = dataset_meta.to_pandas()
        
        # CRITICAL: Track original indices
        df_all['orig_index'] = df_all.index 
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if 'binary_label' not in df_all.columns:
        df_all['binary_label'] = df_all['dataset_type'].apply(lambda x: 1 if 'mental_health' in str(x) else 0)

    # 2. Split Train/Test by AUTHOR
    print("Splitting by Author...")
    valid_authors = df_all[~df_all['author'].isin(['unknown', '[deleted]', ''])]
    unique_authors = valid_authors[['author', 'binary_label']].drop_duplicates()
    
    train_authors, test_authors = train_test_split(
        unique_authors, 
        test_size=0.15, 
        stratify=unique_authors['binary_label'], 
        random_state=SEED
    )
    
    train_mask = df_all['author'].isin(train_authors['author'])
    unknown_mask = ~df_all['author'].isin(valid_authors['author'])
    train_mask = train_mask | unknown_mask
    
    train_df = df_all[train_mask].copy()
    test_df = df_all[~train_mask].copy()
    
    print(f"Train Size: {len(train_df)}")
    print(f"Test Size:  {len(test_df)}")

    # ==========================================================
    # PHASE 1: FILTER RISK
    # ==========================================================
    print("\n--- Phase 1: Filtering Risk Data ---")
    risk_df = train_df[train_df['binary_label'] == 1].copy()
    
    print("Applying Emotion Filter to Risk...")
    risk_df['pos_score'] = risk_df.apply(lambda x: get_emotion_score(x, 'positive'), axis=1)
    risk_df['neg_score'] = risk_df.apply(lambda x: get_emotion_score(x, 'negative'), axis=1)
    
    mask_bad_risk = (risk_df['pos_score'] > HIGH_EMOTION_THRESHOLD) & (risk_df['neg_score'] < LOW_EMOTION_THRESHOLD)
    risk_df_clean_emotion = risk_df[~mask_bad_risk].copy()
    print(f"Removed {mask_bad_risk.sum()} 'Happy Risk' items.")
    
    if len(risk_df_clean_emotion) == 0:
        risk_df_clean_emotion = risk_df.copy()

    print(f"Calculating Risk Self-Density (N={len(risk_df_clean_emotion)})...")
    
    # FETCH EMBEDDINGS BY INDEX
    risk_indices = risk_df_clean_emotion['orig_index'].values
    clean_risk_vecs = all_embeddings[risk_indices]
    
    if len(clean_risk_vecs) > 0:
        risk_density = calculate_density_faiss(clean_risk_vecs, clean_risk_vecs, k=NEIGHBOR_K)
        risk_df_clean_emotion['density'] = risk_density
        clean_risk_df = risk_df_clean_emotion[risk_df_clean_emotion['density'] > RISK_DENSITY_THRESHOLD].copy()
        print(f"Clean Risk (Density > {RISK_DENSITY_THRESHOLD}): {len(risk_df)} -> {len(clean_risk_df)}")
    else:
        clean_risk_df = risk_df.copy()
        print("Warning: No risk data left after emotion filtering.")

    # ==========================================================
    # PHASE 2: SAFE SAMPLING
    # ==========================================================
    print("\n--- Phase 2: Safe Sampling & Hard Negative Mining ---")
    safe_df = train_df[train_df['binary_label'] == 0].copy()
    
    # FETCH EMBEDDINGS BY INDEX
    safe_indices = safe_df['orig_index'].values
    safe_vecs = all_embeddings[safe_indices]
    
    print(f"Calculating Safe Self-Density (N={len(safe_df)})...")
    safe_density = calculate_density_faiss(safe_vecs, safe_vecs, k=NEIGHBOR_K)
    safe_df['density'] = safe_density
    
    safe_df['neg_score'] = safe_df.apply(lambda x: get_emotion_score(x, 'negative'), axis=1)
    safe_df['pos_score'] = safe_df.apply(lambda x: get_emotion_score(x, 'positive'), axis=1)
    
    sad_safe = safe_df[safe_df['neg_score'] > 0.5].copy()
    happy_safe = safe_df[safe_df['pos_score'] > 0.5].copy()
    
    special_indices = set(sad_safe.index) | set(happy_safe.index)
    neutral_safe_candidates = safe_df[~safe_df.index.isin(special_indices)]
    neutral_safe = neutral_safe_candidates[neutral_safe_candidates['density'] > SAFE_DENSITY_THRESHOLD].copy()
    
    print(f"Safe Breakdown: Sad={len(sad_safe)}, Happy={len(happy_safe)}, Neutral={len(neutral_safe)}")
    
    # Sampling Logic
    risk_count = len(clean_risk_df)
    target_sad = int(risk_count * 1.0) 
    target_happy = int(risk_count * 0.5)

    max_unique_sad = int(target_sad / 3)
    max_unique_happy = int(target_happy / 2)

    if len(sad_safe) > max_unique_sad: sad_safe = sad_safe.sample(n=max_unique_sad, random_state=SEED)
    if len(happy_safe) > max_unique_happy: happy_safe = happy_safe.sample(n=max_unique_happy, random_state=SEED)

    sad_safe_oversampled = pd.concat([sad_safe] * 3)
    happy_safe_oversampled = pd.concat([happy_safe] * 2)
    
    target_neutral = len(clean_risk_df)
    if len(neutral_safe) > target_neutral:
        neutral_safe = neutral_safe.sample(n=target_neutral, random_state=SEED)
    
    final_safe = pd.concat([sad_safe_oversampled, happy_safe_oversampled, neutral_safe])
    final_train = pd.concat([clean_risk_df, final_safe])
    final_train = final_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"Final Training Set: {len(final_train)} rows")

    # ==========================================================
    # PHASE 3: SOFT LABELS
    # ==========================================================
    print("\n--- Phase 3: Generating Soft Labels (Teacher Index) ---")
    if 'subreddit' not in final_train.columns: sys.exit(1)
        
    unique_subreddits = sorted(df_all['subreddit'].unique())
    subreddit_map = {sub: i for i, sub in enumerate(unique_subreddits)}
    
    # Get vectors for the final training set
    train_indices = final_train['orig_index'].values
    train_vecs = all_embeddings[train_indices]
    
    # Get vectors for the test set
    test_indices = test_df['orig_index'].values
    test_vecs = all_embeddings[test_indices]
    
    # Use FULL clean risk dataset as teacher to have broad coverage
    teacher_df = clean_risk_df # We use the risk data as the 'anchor' or 'teacher' often? 
    # Wait, the original logic used final_train as teacher. Let's revert to that to be safe.
    # The previous script used final_train as teacher for both itself (leave-one-out implied? no, just self-training) and test.
    teacher_df = final_train
    teacher_indices = teacher_df['orig_index'].values
    teacher_vecs = all_embeddings[teacher_indices]
    
    print("Generating Train Soft Labels...")
    train_soft = compute_soft_labels_faiss(train_vecs, teacher_vecs, teacher_df, 
                                         n_neighbors=50, subreddit_map=subreddit_map)
    final_train['soft_label'] = list(train_soft)
    
    print("Generating Test Soft Labels...")
    test_soft = compute_soft_labels_faiss(test_vecs, teacher_vecs, teacher_df, 
                                        n_neighbors=50, subreddit_map=subreddit_map)
    test_df['soft_label'] = list(test_soft)

    # ==========================================================
    # SAVE
    # ==========================================================
    print("\n--- Saving Data ---")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    with open(os.path.join(DATA_DIR, "subreddit_mapping.json"), "w") as f: json.dump(subreddit_map, f)
    
    def save_parquet(dataframe, filename):
        out = dataframe.rename(columns={'input': 'text'})
        out['label'] = out['binary_label'].astype(int)
        cols = ['text', 'label', 'soft_label']
        if 'subreddit' in out.columns: cols.append('subreddit')
        # We drop orig_index, density, etc. to save space
        out[cols].to_parquet(filename, index=False)
        print(f"Saved {len(out)} to {filename}")
    
    save_parquet(final_train, TRAIN_FILE)
    save_parquet(test_df, TEST_FILE)
    
    print(f"Saved Train: {TRAIN_FILE}")
    print(f"Saved Test:  {TEST_FILE}")
    print("--- Compilation Complete ---")

if __name__ == "__main__":
    main()
