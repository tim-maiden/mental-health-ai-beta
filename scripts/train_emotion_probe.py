import os
import sys
import json
import ast
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.clients import supabase
from src.config import MODELS_DIR, DATA_DIR

# --- CONFIGURATION ---
GOEMOTIONS_TABLE = "goemotions_embeddings"
SAFETY_TABLE = "reddit_safe_embeddings"
MODEL_PATH = os.path.join(MODELS_DIR, "emotion_probe.pkl")
BINARIZER_PATH = os.path.join(MODELS_DIR, "emotion_binarizer.pkl")
CACHE_PATH = os.path.join(DATA_DIR, "goemotions_cache.pkl")

# --- EMOTION DEFINITIONS (Granular) ---
# High-energy, unambiguous positive emotions
POSITIVE_EMOTIONS = {
    "joy", "love", "excitement", "admiration", "optimism", 
    "pride", "amusement", "gratitude"
}

# Clear negative emotions (excluding subtle ones like confusion)
NEGATIVE_EMOTIONS = {
    "sadness", "grief", "anger", "fear", "nervousness", 
    "remorse", "disgust", "annoyance", "disappointment"
}

# We will train on the UNION of these two sets.
TARGET_EMOTIONS = list(POSITIVE_EMOTIONS.union(NEGATIVE_EMOTIONS))

def process_embedding_str(x):
    """Converts string representation of list to numpy array."""
    if x is None:
        return None
    try:
        if isinstance(x, list):
            return np.array(x, dtype=np.float32)
        # Handle string representation if necessary
        return np.array(ast.literal_eval(str(x)), dtype=np.float32)
    except (ValueError, SyntaxError):
        return None

def fetch_goemotions_data(limit=None, use_cache=True):
    """Fetches embeddings and labels from GoEmotions table, with caching."""
    # Check cache
    if use_cache and os.path.exists(CACHE_PATH) and limit is None:
        print(f"Loading GoEmotions from cache: {CACHE_PATH}")
        try:
            return pd.read_pickle(CACHE_PATH)
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-fetching.")

    print(f"Fetching data from {GOEMOTIONS_TABLE}...")
    
    query = supabase.table(GOEMOTIONS_TABLE).select("embedding, emotions")
    
    # We need to page through results because Supabase limits rows per request
    all_data = []
    batch_size = 1000
    offset = 0
    
    while True:
        try:
            r = query.range(offset, offset + batch_size - 1).execute()
            data = r.data
            if not data:
                break
            all_data.extend(data)
            offset += len(data)
            print(f"Fetched {len(all_data)} rows...", end="\r")
            
            if limit and len(all_data) >= limit:
                break
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    print(f"\nTotal GoEmotions rows: {len(all_data)}")
    df = pd.DataFrame(all_data)
    
    # Process embeddings
    print("Processing embeddings...")
    df['embedding_vec'] = df['embedding'].apply(process_embedding_str)
    
    # Process emotions (ensure they are lists)
    print("Processing labels...")
    def parse_emotions(x):
        if isinstance(x, str):
            try:
                loaded = json.loads(x)
                if isinstance(loaded, str):
                     try:
                         return json.loads(loaded)
                     except:
                         return [loaded] if loaded else []
                return loaded if isinstance(loaded, list) else []
            except:
                try:
                    return ast.literal_eval(x)
                except:
                    return []
        return x if isinstance(x, list) else []
        
    df['emotions_list'] = df['emotions'].apply(parse_emotions)
    
    # --- GRANULAR FILTERING ---
    print("Filtering for Target Granular Emotions...")
    target_set = set(TARGET_EMOTIONS)
    
    def filter_emotions(emotions_list):
        # Keep only emotions in our target list
        return [e for e in emotions_list if e in target_set]
        
    df['emotions_list'] = df['emotions_list'].apply(filter_emotions)
    
    # Drop rows with no relevant emotions
    df = df[df['emotions_list'].map(len) > 0]
    
    # Drop invalid rows
    df = df.dropna(subset=['embedding_vec'])
    
    print(f"Filtered DataFrame Size: {len(df)} rows")
    
    # Save to cache if full fetch
    if use_cache and limit is None:
        print(f"Saving cache to {CACHE_PATH}...")
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        df.to_pickle(CACHE_PATH)
        
    return df

def train_probe(df):
    """Trains a multi-label Logistic Regression classifier on granular emotions."""
    print("Preparing training data...")
    
    # 1. Extract Labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['emotions_list'])
    print(f"Classes found ({len(mlb.classes_)}): {mlb.classes_}")
    
    # 2. Extract Features
    X = np.stack(df['embedding_vec'].values)
    
    # 3. Aggressively Free Memory
    print("Freeing DataFrame memory...")
    del df
    import gc
    gc.collect()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    del X
    del y
    gc.collect()

    # --- Standardize ---
    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training Data Shape: {X_train.shape}")
    print("Training Granular Logistic Regression (OneVsRest)...")
    
    # Use OneVsRest with Logistic Regression
    base_clf = LogisticRegression(
        solver='lbfgs',
        class_weight=None,
        max_iter=1000,
        random_state=42,
        verbose=1
    )
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    print("Accuracy (Subset accuracy):", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))
    
    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    with open(BINARIZER_PATH, 'wb') as f:
        pickle.dump(mlb, f)
    SCALER_PATH = os.path.join(MODELS_DIR, "emotion_scaler.pkl")
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Model saved to {MODEL_PATH}")
    
    return clf, mlb

def process_safety_data_in_batches(clf, mlb, batch_size=2000, limit=None, scaler=None):
    """Fetches IDs and embeddings, predicts granular probs, aggregates (Late Fusion), and updates."""
    print(f"Processing data from {SAFETY_TABLE} in batches (Cursor Pagination / Late Fusion)...")
    
    # Get indices for our buckets
    pos_indices = [i for i, label in enumerate(mlb.classes_) if label in POSITIVE_EMOTIONS]
    neg_indices = [i for i, label in enumerate(mlb.classes_) if label in NEGATIVE_EMOTIONS]
    
    print(f"Positive Indices: {pos_indices} (Labels: {[mlb.classes_[i] for i in pos_indices]})")
    print(f"Negative Indices: {neg_indices} (Labels: {[mlb.classes_[i] for i in neg_indices]})")

    last_id = 0
    total_processed = 0
    
    # Stats counters
    stats = {"positive": 0, "negative": 0, "neutral": 0}

    def push_update(item):
        try:
            # Update predicted_emotions (array) and emotion_scores (jsonb)
            # We keep emotion_scores for debugging/sampling if needed, though simpler now
            supabase.table(SAFETY_TABLE).update({
                "predicted_emotions": item["predicted_emotions"],
                "emotion_scores": item["emotion_scores"]
            }).eq("id", item["id"]).execute()
            return True
        except Exception as e:
            return False

    while True:
        try:
            print(f"Fetching batch (id > {last_id})...")
            
            # Cursor pagination: Order by ID, filter > last_id
            r = supabase.table(SAFETY_TABLE)\
                .select("id, embedding")\
                .order("id", desc=False)\
                .gt("id", last_id)\
                .limit(batch_size)\
                .execute()
                
            data = r.data
            
            if not data:
                print("No more data returned from Supabase.")
                break
                
            batch_df = pd.DataFrame(data)
            
            # Update Cursor immediately for next loop
            last_id = batch_df['id'].max()
            
            batch_df['embedding_vec'] = batch_df['embedding'].apply(process_embedding_str)
            batch_df = batch_df.dropna(subset=['embedding_vec'])
            
            if not batch_df.empty:
                X_batch = np.stack(batch_df['embedding_vec'].values)
                if scaler:
                    X_batch = scaler.transform(X_batch)
                
                # Predict raw probabilities for all 17+ classes
                probas = clf.predict_proba(X_batch)
                
                # --- LATE FUSION AGGREGATION ---
                # Use np.max instead of np.sum to enforce single-emotion confidence
                pos_scores = np.max(probas[:, pos_indices], axis=1)
                neg_scores = np.max(probas[:, neg_indices], axis=1)
                
                # Apply Thresholds (High purity requirement)
                THRESHOLD = 0.90
                
                updates = []
                
                for idx, row in batch_df.iterrows():
                    labels = []
                    p_score = pos_scores[idx]
                    n_score = neg_scores[idx]
                    
                    if p_score >= THRESHOLD:
                        labels.append("positive")
                        stats["positive"] += 1
                    if n_score >= THRESHOLD:
                        labels.append("negative")
                        stats["negative"] += 1
                    
                    if not labels:
                        stats["neutral"] += 1

                    # Store scores for "Top N" sampling later
                    scores = {
                        "positive": float(p_score),
                        "negative": float(n_score)
                    }
                    
                    updates.append({
                        "id": row['id'],
                        "predicted_emotions": labels,
                        "emotion_scores": scores
                    })
                
                print(f"  > Uploading predictions for {len(updates)} rows...")
                with ThreadPoolExecutor(max_workers=10) as executor:
                    list(executor.map(push_update, updates))
            
            total_processed += len(data)
            
            import gc
            del data
            del batch_df
            gc.collect()
            
            if limit and total_processed >= limit:
                print(f"Hit limit of {limit} rows.")
                break
                
        except Exception as e:
            print(f"Error in batch processing: {e}")
            break
            
    print(f"\nFinal Batch Stats: Positive: {stats['positive']}, Negative: {stats['negative']}, Neutral/Ambiguous: {stats['neutral']}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the probe model")
    parser.add_argument("--predict", action="store_true", help="Predict on safety data")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--no-cache", action="store_true", help="Force re-fetch from Supabase")
    args = parser.parse_args()
    
    # TRAIN PHASE
    clf, mlb = None, None
    
    if args.train:
        df_go = fetch_goemotions_data(limit=args.limit, use_cache=not args.no_cache)
        clf, mlb = train_probe(df_go)
        # df_go is deleted inside train_probe now
        import gc
        gc.collect()
        
    # PREDICT PHASE
    if args.predict:
        print("WARNING: Prediction tables may not be ready. Proceeding...")
        # Load model if not trained in this run
        if clf is None:
            SCALER_PATH = os.path.join(MODELS_DIR, "emotion_scaler.pkl")
            
            if os.path.exists(MODEL_PATH) and os.path.exists(BINARIZER_PATH):
                print("Loading saved model...")
                with open(MODEL_PATH, 'rb') as f:
                    clf = pickle.load(f)
                with open(BINARIZER_PATH, 'rb') as f:
                    mlb = pickle.load(f)
                
                # Check for scaler
                if os.path.exists(SCALER_PATH):
                    with open(SCALER_PATH, 'rb') as f:
                        scaler = pickle.load(f)
                    print("Loaded scaler.")
                else:
                    scaler = None
            else:
                print("Model not found. Please run with --train first.")
                return
        
        # Stream processing instead of loading all
        process_safety_data_in_batches(clf, mlb, batch_size=1000, limit=args.limit, scaler=scaler)

if __name__ == "__main__":
    main()
