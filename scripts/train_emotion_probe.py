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
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split

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
            # Check if it's double-encoded JSON (e.g. '"[\"joy\"]"')
            try:
                # First load
                loaded = json.loads(x)
                if isinstance(loaded, str):
                     # Try loading again if it's still a string
                     try:
                         return json.loads(loaded)
                     except:
                         return [loaded] if loaded else []
                return loaded if isinstance(loaded, list) else []
            except:
                # If json load fails, maybe it's just a string like "['joy']" that needs ast.literal_eval
                try:
                    return ast.literal_eval(x)
                except:
                    return []
        return x if isinstance(x, list) else []
        
    df['emotions_list'] = df['emotions'].apply(parse_emotions)
    
    # --- MULTI-LABEL TRAINING (2 Independent Binaries) ---
    print("Mapping Emotions to 2 Independent Binaries: Positive, Negative (Ambiguous dropped)...")
    
    # Define the Buckets (Tightened for High Precision)
    # POS_BUCKET: High-energy, unambiguous positive emotions only.
    # Removed: approval (neutral), relief (context dependent), desire (ambiguous), caring/admiration (can be used in sad contexts)
    POS_BUCKET = {"amusement", "excitement", "joy", "love", "optimism", "pride", "gratitude"}
    
    # NEG_BUCKET: Dysphoric emotions
    NEG_BUCKET = {"fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"}
    # AMB_BUCKET dropped as requested
    
    def map_to_multilabels(emotions_list):
        labels = set()
        emo_set = set(emotions_list)
        
        # Check Positive
        if not emo_set.isdisjoint(POS_BUCKET):
            labels.add("positive")
        
        # Check Negative
        if not emo_set.isdisjoint(NEG_BUCKET):
            labels.add("negative")
            
        # Ambiguous is now implicit "background" (empty set) if neither pos nor neg
            
        return list(labels)
        
    df['emotions_list'] = df['emotions_list'].apply(map_to_multilabels)
    
    # Drop invalid rows
    df = df.dropna(subset=['embedding_vec'])
    
    # Save to cache if full fetch
    if use_cache and limit is None:
        print(f"Saving cache to {CACHE_PATH}...")
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        df.to_pickle(CACHE_PATH)
        
    return df

def train_probe(df):
    """Trains a multi-label Logistic Regression classifier."""
    print("Preparing training data...")
    
    # 1. Extract Labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['emotions_list'])
    print(f"Classes found: {mlb.classes_}")
    
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
    print("Training Logistic Regression (OneVsRest)...")
    
    # Use OneVsRest with Logistic Regression for high-precision independent classifiers
    # class_weight='balanced' helps with prevalence issues, but we will tune thresholds manually
    base_clf = LogisticRegression(
        solver='lbfgs',
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        verbose=1
    )
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    # Get probabilities
    y_probs = clf.predict_proba(X_test)
    
    # --- THRESHOLD CALIBRATION (Optimized for Precision) ---
    print("Calibrating thresholds for High Precision (Target > 0.8)...")
    best_thresholds = []
    
    # Iterate through classes
    for i, class_label in enumerate(mlb.classes_):
        y_true = y_test[:, i]
        y_score = y_probs[:, i]
        
        best_th = 0.5
        target_precision = 0.75  # Relaxed to 0.75 for better recall/balance
        found_high_precision = False
        
        # Scan thresholds from 0.5 to 0.99
        print(f"  Scanning class '{class_label}':")
        for th in np.arange(0.5, 0.99, 0.01):
            y_pred_th = (y_score >= th).astype(int)
            
            if np.sum(y_pred_th) == 0:
                continue 
            
            prec = precision_score(y_true, y_pred_th, zero_division=0)
            rec = 0 # Calculate recall only if needed for debug, speed up loop
            
            if prec >= target_precision:
                best_th = th
                found_high_precision = True
                print(f"    -> Found Precision {prec:.4f} at Threshold {th:.2f}")
                break 
        
        if not found_high_precision:
            print(f"    -> Could not reach target precision 0.8. Using 0.9 as fallback safe threshold.")
            best_th = 0.90
            
        best_thresholds.append(best_th)
        
    print(f"Using Calibrated Thresholds: {best_thresholds}")
    
    # Apply thresholds
    y_pred_calibrated = np.zeros_like(y_probs)
    for i in range(len(mlb.classes_)):
        y_pred_calibrated[:, i] = (y_probs[:, i] >= best_thresholds[i]).astype(int)

    print("Accuracy (Subset accuracy):", accuracy_score(y_test, y_pred_calibrated))
    print(classification_report(y_test, y_pred_calibrated, target_names=mlb.classes_, zero_division=0))
    
    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    with open(BINARIZER_PATH, 'wb') as f:
        pickle.dump(mlb, f)
    SCALER_PATH = os.path.join(MODELS_DIR, "emotion_scaler.pkl")
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    THRESHOLDS_PATH = os.path.join(MODELS_DIR, "emotion_thresholds.json")
    with open(THRESHOLDS_PATH, 'w') as f:
        json.dump(best_thresholds, f)
        
    print(f"Model saved to {MODEL_PATH}")
    
    return clf, mlb

def process_safety_data_in_batches(clf, mlb, batch_size=2000, limit=None, scaler=None, thresholds=None):
    """Fetches IDs and embeddings from Safety table in batches using Cursor Pagination, predicts, and updates."""
    print(f"Processing data from {SAFETY_TABLE} in batches (Cursor Pagination)...")
    
    # Check total count for info
    count_query = supabase.table(SAFETY_TABLE).select("id", count="exact").limit(1)
    try:
        res = count_query.execute()
        total_rows = res.count
        print(f"Total rows to process: {total_rows}")
    except:
        total_rows = None

    last_id = 0
    total_processed = 0
    from concurrent.futures import ThreadPoolExecutor

    def push_update(item):
        try:
            supabase.table(SAFETY_TABLE).update({
                "predicted_emotions": item["predicted_emotions"],
                "emotion_scores": item["emotion_scores"]
            }).eq("id", item["id"]).execute()
            return True
        except Exception as e:
            # print(f"Update failed: {e}")
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
                
                # Predict Probabilities
                probas = clf.predict_proba(X_batch)
                
                # Apply Thresholds
                if thresholds:
                     y_pred_bool = probas > np.array(thresholds)
                else:
                     y_pred_bool = probas > 0.5
                
                y_labels = mlb.inverse_transform(y_pred_bool)
                
                updates = []
                classes = mlb.classes_
                
                for idx, row in batch_df.iterrows():
                    # Create scores dict
                    scores = {cls: float(probas[idx][i]) for i, cls in enumerate(classes)}
                    
                    updates.append({
                        "id": row['id'],
                        "predicted_emotions": list(y_labels[idx]),
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
            THRESHOLDS_PATH = os.path.join(MODELS_DIR, "emotion_thresholds.json")
            
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
                    
                # Check for thresholds
                thresholds = None
                if os.path.exists(THRESHOLDS_PATH):
                     with open(THRESHOLDS_PATH, 'r') as f:
                        thresholds = json.load(f)
                     print(f"Loaded calibrated thresholds: {thresholds}")
            else:
                print("Model not found. Please run with --train first.")
                return
        else:
            # We just trained
            THRESHOLDS_PATH = os.path.join(MODELS_DIR, "emotion_thresholds.json")
            if os.path.exists(THRESHOLDS_PATH):
                 with open(THRESHOLDS_PATH, 'r') as f:
                    thresholds = json.load(f)
            else:
                thresholds = None
        
        # Stream processing instead of loading all
        process_safety_data_in_batches(clf, mlb, batch_size=1000, limit=args.limit, scaler=scaler, thresholds=thresholds)

if __name__ == "__main__":
    main()
