import os
import sys
import pandas as pd
import numpy as np
import json
import pyarrow.parquet as pq
import s3fs
from sklearn.model_selection import train_test_split

# Import FAISS (GPU-accelerated k-NN)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: faiss not installed. Please install faiss-gpu (or faiss-cpu).")
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
# from src.data.storage import fetch_data_parallel

# --- CONFIGURATION ---
# (Moved to config.py)

def calculate_density_faiss(query_vecs, ref_vecs, k=NEIGHBOR_K):
    """
    Calculates the average cosine similarity to the k nearest neighbors using FAISS (GPU if available).
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not installed. Cannot calculate density.")

    d = ref_vecs.shape[1]
    
    # Ensure float32
    query_vecs = query_vecs.astype(np.float32)
    ref_vecs = ref_vecs.astype(np.float32)
    
    # L2 Normalize vectors to ensure Inner Product == Cosine Similarity
    faiss.normalize_L2(query_vecs)
    faiss.normalize_L2(ref_vecs) 
    
    print(f"Build FAISS Index (d={d}, ref_size={len(ref_vecs)})...")
    
    # Setup Index (Inner Product = Cosine if normalized)
    index = faiss.IndexFlatIP(d)
    
    # GPU Transfer if available
    try:
        res = faiss.StandardGpuResources()
        print("Using FAISS GPU...")
        index = faiss.index_cpu_to_gpu(res, 0, index)
    except Exception as e:
        print(f"FAISS GPU not available ({e}). Switching to fast CPU index (HNSW)...")
        # Use HNSW for faster CPU search (approximate but much faster than Flat)
        # HNSW32 with Inner Product
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.train(ref_vecs) # HNSW sometimes needs training if stacked, usually not for Flat, but good practice
        
    index.add(ref_vecs)
    
    print(f"Searching Index (query_size={len(query_vecs)}, k={k})...")
    D, I = index.search(query_vecs, k)
    
    densities = np.mean(D, axis=1)
    
    return densities

def compute_soft_labels_faiss(query_df, teacher_df, n_neighbors=50, temperature=1.0, subreddit_map=None):
    """
    Computes soft labels using Weighted k-NN with FAISS.
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not installed.")

    print(f"Computing k-NN Soft Labels (k={n_neighbors}, temp={temperature})...")
    
    query_vecs = np.stack(query_df['embedding_vec'].values).astype(np.float32)
    teacher_vecs = np.stack(teacher_df['embedding_vec'].values).astype(np.float32)
    
    # L2 Normalize for Cosine Similarity
    faiss.normalize_L2(query_vecs)
    faiss.normalize_L2(teacher_vecs)
    
    teacher_subs = teacher_df['subreddit'].map(subreddit_map).values.astype(int)
    num_classes = len(subreddit_map)
    d = teacher_vecs.shape[1]

    index = faiss.IndexFlatIP(d)
    try:
        res = faiss.StandardGpuResources()
        print("Using FAISS GPU...")
        index = faiss.index_cpu_to_gpu(res, 0, index)
    except Exception as e:
        print(f"FAISS GPU not available ({e}). Switching to fast CPU index (HNSW)...")
        # Use HNSW for faster CPU search
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.train(teacher_vecs)
        
    index.add(teacher_vecs)
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
    # Fallback if predicted_emotions list exists (legacy)
    ems = row.get('predicted_emotions', [])
    if isinstance(ems, list) and target in ems: return 1.0
    return 0.0

def main():
    print("--- Starting GPU-Accelerated Dataset Compilation (Abundance Strategy) ---")
    
    if not FAISS_AVAILABLE:
        print("Error: FAISS is required. Install faiss-gpu or faiss-cpu.")
        sys.exit(1)
        
    # 1. Load Data
    print(f"Loading data via S3 Parquet Snapshot: {RAW_DATA_FILE}...")

    try:
        # SETUP S3 FILESYSTEM
        fs = s3fs.S3FileSystem(
            key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
            client_kwargs={"region_name": os.getenv("AWS_REGION", "us-east-1")}
        )
        
        # PARSE S3 PATH
        # Remove s3:// prefix if present, as s3fs expects 'bucket/key'
        s3_path = RAW_DATA_FILE.replace("s3://", "")
        
        # READ AS ARROW TABLE (Efficient, Zero-Copy)
        table = pq.read_table(s3_path, filesystem=fs)
        print(f"Loaded Arrow Table: {table.num_rows} rows.")

        # EFFICIENT EMBEDDING CONVERSION (Arrow -> Numpy Matrix)
        # This avoids creating 3.7 billion Python float objects (~100GB RAM)
        print("Reconstructing embedding vectors (High-Performance Mode)...")
        
        # Extract flat values from the ListArray
        emb_column = table['embedding']
        
        # FIX: Do NOT use combine_chunks() on the ListArray itself.
        # The total element count (~3.7B) exceeds the 32-bit offset limit (2.1B) of PyArrow ListArrays.
        # Instead, we process each chunk individually and flatten them to Numpy (which handles large data fine).
        print(f"Flattening {len(emb_column.chunks)} chunks manually to bypass offset limits...")
        
        chunk_arrays = []
        for chunk in emb_column.chunks:
            # .flatten() handles slicing correctly and returns the raw flat values for this chunk
            chunk_arrays.append(chunk.flatten().to_numpy())
            
        # Concatenate the raw float data (High RAM usage momentarily, but safe from overflow)
        flattened_data = np.concatenate(chunk_arrays)
        matrix = flattened_data.reshape(-1, 1536)
        
        # CONVERT METADATA TO PANDAS
        # We drop the heavy 'embedding' column before converting to Pandas
        metadata_cols = [c for c in table.column_names if c != 'embedding']
        df_all = table.select(metadata_cols).to_pandas()
        
        # ASSIGN EMBEDDINGS
        # Create a Series of numpy arrays (much lighter than lists of floats)
        df_all['embedding_vec'] = list(matrix)
        
        print(f"Loaded {len(df_all)} rows successfully.")

    except Exception as e:
        print(f"Error loading S3 snapshot: {e}")
        # print("Tip: Run 'python scripts/ingest_data.py' to generate the snapshot first.") # Optional tip
        sys.exit(1)

    # Ensure binary label exists (1 if risk, 0 if safe)
    # df_all should already have 'dataset_type' from ingest_data.py
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
    # PHASE 1: FILTER RISK (High Purity)
    # ==========================================================
    print("\n--- Phase 1: Filtering Risk Data ---")
    
    risk_df = train_df[train_df['binary_label'] == 1].copy()
    risk_vecs = np.stack(risk_df['embedding_vec'].values)
    
    # 1. Emotion Filtering (Remove "Happy Risk")
    print("Applying Emotion Filter to Risk...")
    risk_df['pos_score'] = risk_df.apply(lambda x: get_emotion_score(x, 'positive'), axis=1)
    risk_df['neg_score'] = risk_df.apply(lambda x: get_emotion_score(x, 'negative'), axis=1)
    
    # Remove if (High Positive AND Low Negative)
    mask_bad_risk = (risk_df['pos_score'] > HIGH_EMOTION_THRESHOLD) & (risk_df['neg_score'] < LOW_EMOTION_THRESHOLD)
    risk_df_clean_emotion = risk_df[~mask_bad_risk].copy()
    print(f"Removed {mask_bad_risk.sum()} 'Happy Risk' items (Pos > {HIGH_EMOTION_THRESHOLD} & Neg < {LOW_EMOTION_THRESHOLD}).")
    
    if len(risk_df_clean_emotion) == 0:
        print("Warning: All risk items removed by emotion filter! Reverting to original risk set (check data/thresholds).")
        risk_df_clean_emotion = risk_df.copy()

    # 2. Density Filtering (Self-Density)
    print(f"Calculating Risk Self-Density (N={len(risk_df_clean_emotion)})...")
    clean_risk_vecs = np.stack(risk_df_clean_emotion['embedding_vec'].values)
    risk_density = calculate_density_faiss(clean_risk_vecs, clean_risk_vecs, k=NEIGHBOR_K)
    risk_df_clean_emotion['density'] = risk_density
    
    clean_risk_df = risk_df_clean_emotion[risk_df_clean_emotion['density'] > RISK_DENSITY_THRESHOLD].copy()
    print(f"Clean Risk (Density > {RISK_DENSITY_THRESHOLD}): {len(risk_df)} -> {len(clean_risk_df)}")

    # ==========================================================
    # PHASE 2: SAFE SAMPLING (Sad/Happy/Neutral)
    # ==========================================================
    print("\n--- Phase 2: Safe Sampling & Hard Negative Mining ---")
    
    safe_df = train_df[train_df['binary_label'] == 0].copy()
    safe_vecs = np.stack(safe_df['embedding_vec'].values)
    
    print(f"Calculating Safe Self-Density (N={len(safe_df)})...")
    safe_density = calculate_density_faiss(safe_vecs, safe_vecs, k=NEIGHBOR_K)
    safe_df['density'] = safe_density
    
    safe_df['neg_score'] = safe_df.apply(lambda x: get_emotion_score(x, 'negative'), axis=1)
    safe_df['pos_score'] = safe_df.apply(lambda x: get_emotion_score(x, 'positive'), axis=1)
    
    # 1. Sad Safe (Hard Negatives)
    sad_safe = safe_df[safe_df['neg_score'] > 0.5].copy()
    
    # 2. Happy Safe
    happy_safe = safe_df[safe_df['pos_score'] > 0.5].copy()
    
    # 3. Neutral Safe (High Density Safe)
    special_indices = set(sad_safe.index) | set(happy_safe.index)
    neutral_safe_candidates = safe_df[~safe_df.index.isin(special_indices)]
    neutral_safe = neutral_safe_candidates[neutral_safe_candidates['density'] > SAFE_DENSITY_THRESHOLD].copy()
    
    print(f"Safe Breakdown:")
    print(f" - Sad Safe (Neg > 0.5): {len(sad_safe)}")
    print(f" - Happy Safe (Pos > 0.5): {len(happy_safe)}")
    print(f" - Neutral Safe (Density > {SAFE_DENSITY_THRESHOLD}): {len(neutral_safe)}")
    
    # Oversampling Strategy
    print("Applying Adjusted Oversampling Multipliers...")
    
    # Oversample Sad/Happy safe examples to improve decision boundaries against Risk.
    sad_safe_oversampled = pd.concat([sad_safe] * 3)
    happy_safe_oversampled = pd.concat([happy_safe] * 2)
    
    print(f" - Sad Safe (3x): {len(sad_safe_oversampled)}")
    print(f" - Happy Safe (2x): {len(happy_safe_oversampled)}")
    
    # CAP NEUTRAL SAFE
    # Cap neutral safe samples to match Risk count for class balance.
    target_neutral_count = len(clean_risk_df)
    if len(neutral_safe) > target_neutral_count:
        print(f" - Capping Neutral Safe ({len(neutral_safe)} -> {target_neutral_count})...")
        neutral_safe = neutral_safe.sample(n=target_neutral_count, random_state=SEED)
    
    final_safe = pd.concat([
        sad_safe_oversampled,
        happy_safe_oversampled,
        neutral_safe
    ])
    
    final_train = pd.concat([clean_risk_df, final_safe])
    final_train = final_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"\nFinal Training Set: {len(final_train)} rows")
    print(f" - Risk: {len(clean_risk_df)}")
    print(f" - Safe: {len(final_safe)}")
    
    # ==========================================================
    # PHASE 3: SOFT LABELS
    # ==========================================================
    print("\n--- Phase 3: Generating Soft Labels (Teacher Index) ---")
    
    if 'subreddit' not in final_train.columns:
        print("Error: 'subreddit' missing.")
        sys.exit(1)
        
    unique_subreddits = sorted(df_all['subreddit'].unique())
    subreddit_map = {sub: i for i, sub in enumerate(unique_subreddits)}
    
    soft_labels_train = compute_soft_labels_faiss(
        query_df=final_train,
        teacher_df=final_train,
        n_neighbors=50,
        temperature=1.0,
        subreddit_map=subreddit_map
    )
    final_train['soft_label'] = soft_labels_train
    
    soft_labels_test = compute_soft_labels_faiss(
        query_df=test_df,
        teacher_df=final_train,
        n_neighbors=50,
        temperature=1.0,
        subreddit_map=subreddit_map
    )
    test_df['soft_label'] = soft_labels_test
    
    mapping_file = os.path.join(DATA_DIR, "subreddit_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump(subreddit_map, f)
        
    risk_subs = set(clean_risk_df['subreddit'].unique())
    risk_indices = [idx for sub, idx in subreddit_map.items() if sub in risk_subs]
    with open(os.path.join(DATA_DIR, "risk_indices.json"), "w") as f:
        json.dump(risk_indices, f)
        
    # ==========================================================
    # EXPORT
    # ==========================================================
    print("\n--- Exporting ---")
    
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
