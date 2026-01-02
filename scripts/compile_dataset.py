import os
import sys
import pandas as pd
import numpy as np
import json
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

# Import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: faiss not installed. Please install faiss-gpu-cu12.")
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
    Prioritizes GPU (H100) -> Falls back to CPU (HNSW) if GPU fails.
    """
    try:
        # 1. Attempt GPU Index (Flat Inner Product = Cosine if normalized)
        # H100 requires faiss-gpu-cu12 to work correctly
        res = faiss.StandardGpuResources() 
        index_cpu = faiss.IndexFlatIP(d)
        
        print(f"  > Moving index to GPU (H100)...")
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        index.add(ref_vecs)
        print("  > GPU Index built successfully.")
        return index

    except Exception as e:
        print(f"\n  [WARNING] GPU Index failed: {e}")
        print("  [INFO] Likely architecture mismatch or missing faiss-gpu-cu12.")
        print("  > Falling back to CPU (IndexHNSWFlat)...")
        
        # 2. CPU Fallback (HNSW for speed)
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.train(ref_vecs)
        index.add(ref_vecs)
        return index

def calculate_density_faiss(query_vecs, ref_vecs, k=NEIGHBOR_K):
    """
    Calculates the average cosine similarity to the k nearest neighbors.
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not installed.")

    d = ref_vecs.shape[1]
    
    # Ensure float32
    query_vecs = query_vecs.astype(np.float32)
    ref_vecs = ref_vecs.astype(np.float32)
    
    # L2 Normalize vectors to ensure Inner Product == Cosine Similarity
    faiss.normalize_L2(query_vecs)
    faiss.normalize_L2(ref_vecs) 
    
    # Get Index (GPU or CPU)
    index = get_faiss_index(d, ref_vecs)
            
    print(f"  > Searching Index (query_size={len(query_vecs)}, k={k})...")
    D, I = index.search(query_vecs, k)
    
    densities = np.mean(D, axis=1)
    return densities

def compute_soft_labels_faiss(query_df, teacher_df, n_neighbors=50, temperature=1.0, subreddit_map=None):
    """
    Computes soft labels using Weighted k-NN.
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not installed.")

    print(f"Computing k-NN Soft Labels (k={n_neighbors}, temp={temperature})...")
    
    query_vecs = np.stack(query_df['embedding_vec'].values).astype(np.float32)
    teacher_vecs = np.stack(teacher_df['embedding_vec'].values).astype(np.float32)
    
    # L2 Normalize
    faiss.normalize_L2(query_vecs)
    faiss.normalize_L2(teacher_vecs)
    
    teacher_subs = teacher_df['subreddit'].map(subreddit_map).values.astype(int)
    num_classes = len(subreddit_map)
    d = teacher_vecs.shape[1]

    # Get Index (GPU or CPU)
    index = get_faiss_index(d, teacher_vecs)
    
    distances, indices = index.search(query_vecs, n_neighbors)
    
    soft_labels = []
    
    # weights = 1 / (1 - sim)
    similarities = distances
    weights = 1.0 / (1.0 - similarities + 1e-6)
    
    neighbor_classes = teacher_subs[indices]
    
    for i in range(len(query_vecs)):
        w = weights[i]
        classes = neighbor_classes[i]
        
        class_scores = np.zeros(num_classes)
        np.add.at(class_scores, classes, w)
        
        logits = class_scores / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        soft_labels.append(probs.tolist())
        
    return soft_labels

def get_emotion_score(row, target):
    """Helper to safely extract emotion scores."""
    scores = row.get('emotion_scores', {})
    if isinstance(scores, str):
        try: scores = json.loads(scores)
        except: scores = {}
    if isinstance(scores, dict):
        return float(scores.get(target, 0.0))
    ems = row.get('predicted_emotions', [])
    if isinstance(ems, list) and target in ems: return 1.0
    return 0.0

def main():
    print("--- Starting Dataset Compilation (GPU Accelerated) ---")
    
    if not FAISS_AVAILABLE:
        print("Error: FAISS is required. Install faiss-gpu-cu12.")
        sys.exit(1)
        
    # 1. Load Data
    print(f"Loading data from Hugging Face Hub (tim-maiden/mental-health-ai)...")
    from datasets import load_dataset

    try:
        dataset = load_dataset("tim-maiden/mental-health-ai", split="train")
        print(f"Loaded Dataset: {len(dataset)} rows.")

        print("Reconstructing embedding vectors (High-Performance Mode)...")
        table = dataset.data
        emb_column = table['embedding']
        
        print(f"Flattening {len(emb_column.chunks)} chunks manually to bypass offset limits...")
        chunk_arrays = []
        for chunk in emb_column.chunks:
            chunk_arrays.append(chunk.flatten().to_numpy())
            
        flattened_data = np.concatenate(chunk_arrays)
        matrix = flattened_data.reshape(-1, 1536)
        
        metadata_cols = [c for c in table.column_names if c != 'embedding']
        df_all = table.select(metadata_cols).to_pandas()
        df_all['embedding_vec'] = list(matrix)
        print(f"Loaded {len(df_all)} rows successfully.")

    except Exception as e:
        print(f"Error loading dataset: {e}")
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
    test_mask = df_all['author'].isin(test_authors['author'])
    unknown_mask = ~df_all['author'].isin(valid_authors['author'])
    train_mask = train_mask | unknown_mask
    
    train_df = df_all[train_mask].copy()
    test_df = df_all[test_mask].copy()
    
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
    clean_risk_vecs = np.stack(risk_df_clean_emotion['embedding_vec'].values)
    risk_density = calculate_density_faiss(clean_risk_vecs, clean_risk_vecs, k=NEIGHBOR_K)
    risk_df_clean_emotion['density'] = risk_density
    
    clean_risk_df = risk_df_clean_emotion[risk_df_clean_emotion['density'] > RISK_DENSITY_THRESHOLD].copy()
    print(f"Clean Risk (Density > {RISK_DENSITY_THRESHOLD}): {len(risk_df)} -> {len(clean_risk_df)}")

    # ==========================================================
    # PHASE 2: SAFE SAMPLING
    # ==========================================================
    print("\n--- Phase 2: Safe Sampling & Hard Negative Mining ---")
    safe_df = train_df[train_df['binary_label'] == 0].copy()
    safe_vecs = np.stack(safe_df['embedding_vec'].values)
    
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
    target_sad = risk_count * 2
    target_happy = risk_count * 1

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
    
    soft_labels_train = compute_soft_labels_faiss(final_train, final_train, 50, 1.0, subreddit_map)
    final_train['soft_label'] = soft_labels_train
    
    soft_labels_test = compute_soft_labels_faiss(test_df, final_train, 50, 1.0, subreddit_map)
    test_df['soft_label'] = soft_labels_test
    
    # ==========================================================
    # EXPORT
    # ==========================================================
    print("\n--- Exporting ---")
    with open(os.path.join(DATA_DIR, "subreddit_mapping.json"), "w") as f: json.dump(subreddit_map, f)
    
    def save_parquet(dataframe, filename):
        out = dataframe.rename(columns={'input': 'text'})
        out['label'] = out['binary_label'].astype(int)
        cols = ['text', 'label', 'soft_label']
        if 'subreddit' in out.columns: cols.append('subreddit')
        out[cols].to_parquet(filename, index=False)
        print(f"Saved {len(out)} to {filename}")
        
    save_parquet(final_train, TRAIN_FILE)
    save_parquet(test_df, TEST_FILE)
    print("Done!")

if __name__ == "__main__":
    main()
