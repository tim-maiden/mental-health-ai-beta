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
from src.config import (
    MODELS_DIR, 
    DATA_DIR,
    POSITIVE_EMOTIONS,
    NEGATIVE_EMOTIONS,
    GOEMOTIONS_TABLE,
    BATCH_SIZE,
    PROBE_CONFIDENCE_THRESHOLD
)

# --- CONFIGURATION ---
# Default table, can be overridden by --table arg
DEFAULT_TARGET_TABLE = "reddit_safe_embeddings"

MODEL_PATH = os.path.join(MODELS_DIR, "emotion_probe.pkl")
BINARIZER_PATH = os.path.join(MODELS_DIR, "emotion_binarizer.pkl")
CACHE_PATH = os.path.join(DATA_DIR, "goemotions_cache.pkl")

# --- EMOTION DEFINITIONS (Granular) ---
TARGET_EMOTIONS = list(POSITIVE_EMOTIONS.union(NEGATIVE_EMOTIONS))

def process_embedding_str(x):
    if x is None: return None
    try:
        if isinstance(x, list): return np.array(x, dtype=np.float32)
        return np.array(ast.literal_eval(str(x)), dtype=np.float32)
    except: return None

def fetch_goemotions_data(limit=None, use_cache=True):
    if use_cache and os.path.exists(CACHE_PATH) and limit is None:
        print(f"Loading GoEmotions from cache: {CACHE_PATH}")
        try: return pd.read_pickle(CACHE_PATH)
        except: print("Cache load failed. Re-fetching.")

    print(f"Fetching data from {GOEMOTIONS_TABLE}...")
    query = supabase.table(GOEMOTIONS_TABLE).select("embedding, emotions")
    all_data = []
    offset = 0
    batch_size = BATCH_SIZE
    
    while True:
        try:
            r = query.range(offset, offset + batch_size - 1).execute()
            if not r.data: break
            all_data.extend(r.data)
            offset += len(r.data)
            print(f"Fetched {len(all_data)} rows...", end="\r")
            if limit and len(all_data) >= limit: break
        except Exception as e:
            print(f"Error: {e}")
            break
            
    print(f"\nTotal GoEmotions rows: {len(all_data)}")
    df = pd.DataFrame(all_data)
    
    print("Processing embeddings...")
    df['embedding_vec'] = df['embedding'].apply(process_embedding_str)
    
    print("Processing labels...")
    def parse_emotions(x):
        if isinstance(x, str):
            try:
                loaded = json.loads(x)
                if isinstance(loaded, str):
                     try: return json.loads(loaded)
                     except: return [loaded] if loaded else []
                return loaded if isinstance(loaded, list) else []
            except:
                try: return ast.literal_eval(x)
                except: return []
        return x if isinstance(x, list) else []
        
    df['emotions_list'] = df['emotions'].apply(parse_emotions)
    
    print("Filtering for Target Granular Emotions...")
    target_set = set(TARGET_EMOTIONS)
    df['emotions_list'] = df['emotions_list'].apply(lambda x: [e for e in x if e in target_set])
    df = df[df['emotions_list'].map(len) > 0]
    df = df.dropna(subset=['embedding_vec'])
    
    if use_cache and limit is None:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        df.to_pickle(CACHE_PATH)
        
    return df

def train_probe(df):
    print("Preparing training data...")
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['emotions_list'])
    X = np.stack(df['embedding_vec'].values)
    
    print(f"Classes found ({len(mlb.classes_)}): {mlb.classes_}")

    del df
    import gc; gc.collect()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Training Granular Logistic Regression (OneVsRest)...")
    # CHANGED: class_weight=None for High Precision
    base_clf = LogisticRegression(solver='lbfgs', class_weight=None, max_iter=1000, random_state=42, verbose=1)
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    print("Accuracy (Subset):", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f: pickle.dump(clf, f)
    with open(BINARIZER_PATH, 'wb') as f: pickle.dump(mlb, f)
    with open(os.path.join(MODELS_DIR, "emotion_scaler.pkl"), 'wb') as f: pickle.dump(scaler, f)
        
    return clf, mlb, scaler

def process_data_in_batches(target_table, clf, mlb, batch_size=BATCH_SIZE, limit=None, scaler=None):
    print(f"Processing data from {target_table} in batches (Cursor Pagination + Late Fusion)...")
    
    # Identify indices for Late Fusion Aggregation
    pos_indices = [i for i, label in enumerate(mlb.classes_) if label in POSITIVE_EMOTIONS]
    neg_indices = [i for i, label in enumerate(mlb.classes_) if label in NEGATIVE_EMOTIONS]
    
    print(f"Positive Buckets: {[mlb.classes_[i] for i in pos_indices]}")
    print(f"Negative Buckets: {[mlb.classes_[i] for i in neg_indices]}")

    last_id = 0
    total_processed = 0
    stats = {"positive": 0, "negative": 0, "neutral": 0}

    def push_update(item):
        try:
            supabase.table(target_table).update({
                "predicted_emotions": item["predicted_emotions"],
                "emotion_scores": item["emotion_scores"]
            }).eq("id", item["id"]).execute()
            return True
        except: return False

    while True:
        try:
            print(f"Fetching batch (id > {last_id})...")
            # Cursor Pagination
            r = supabase.table(target_table)\
                .select("id, embedding")\
                .order("id", desc=False)\
                .gt("id", last_id)\
                .limit(batch_size)\
                .execute()
                
            if not r.data:
                print("No more data returned from Supabase.")
                break
                
            batch_df = pd.DataFrame(r.data)
            
            # Update Cursor
            last_id = batch_df['id'].max()
            
            batch_df['embedding_vec'] = batch_df['embedding'].apply(process_embedding_str)
            batch_df = batch_df.dropna(subset=['embedding_vec'])
            
            if not batch_df.empty:
                X_batch = np.stack(batch_df['embedding_vec'].values)
                if scaler: X_batch = scaler.transform(X_batch)
                
                probas = clf.predict_proba(X_batch)
                
                # --- LATE FUSION: MAX AGGREGATION ---
                pos_scores = np.max(probas[:, pos_indices], axis=1)
                neg_scores = np.max(probas[:, neg_indices], axis=1)
                
                THRESHOLD = PROBE_CONFIDENCE_THRESHOLD
                updates = []
                
                for idx, row in batch_df.iterrows():
                    labels = []
                    p_score = float(pos_scores[idx])
                    n_score = float(neg_scores[idx])
                    
                    if p_score >= THRESHOLD:
                        labels.append("positive")
                        stats["positive"] += 1
                    if n_score >= THRESHOLD:
                        labels.append("negative")
                        stats["negative"] += 1
                    if not labels: stats["neutral"] += 1
                    
                    updates.append({
                        "id": row['id'],
                        "predicted_emotions": labels,
                        "emotion_scores": {"positive": p_score, "negative": n_score}
                    })
                
                print(f"  > Uploading predictions for {len(updates)} rows...")
                with ThreadPoolExecutor(max_workers=10) as executor:
                    list(executor.map(push_update, updates))
            
            total_processed += len(r.data)
            import gc; gc.collect()
            if limit and total_processed >= limit: break
                
        except Exception as e:
            print(f"Error: {e}")
            break
            
    print(f"\nFinal Stats for {target_table}: {stats}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the probe model")
    parser.add_argument("--predict", action="store_true", help="Predict on data")
    parser.add_argument("--table", type=str, default=DEFAULT_TARGET_TABLE, help="Target table for prediction")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows")
    parser.add_argument("--no-cache", action="store_true", help="Force re-fetch GoEmotions")
    args = parser.parse_args()
    
    clf, mlb, scaler = None, None, None
    
    if args.train:
        df_go = fetch_goemotions_data(limit=args.limit, use_cache=not args.no_cache)
        clf, mlb, scaler = train_probe(df_go)
        
    if args.predict:
        if clf is None:
            if os.path.exists(MODEL_PATH):
                print("Loading saved model...")
                with open(MODEL_PATH, 'rb') as f: clf = pickle.load(f)
                with open(BINARIZER_PATH, 'rb') as f: mlb = pickle.load(f)
                scaler_path = os.path.join(MODELS_DIR, "emotion_scaler.pkl")
                scaler = pickle.load(open(scaler_path, 'rb')) if os.path.exists(scaler_path) else None
            else:
                print("Model not found. Run --train first.")
                return
        
        process_data_in_batches(args.table, clf, mlb, limit=args.limit, scaler=scaler)

if __name__ == "__main__":
    main()
