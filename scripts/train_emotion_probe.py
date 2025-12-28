import os
import sys
import json
import ast
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
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
    
    # Drop invalid rows
    df = df.dropna(subset=['embedding_vec'])
    
    # Save to cache if full fetch
    if use_cache and limit is None:
        print(f"Saving cache to {CACHE_PATH}...")
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        df.to_pickle(CACHE_PATH)
        
    return df

def train_probe(df):
    """Trains a multi-label classifier."""
    print("Preparing training data...")
    
    # 1. Extract Labels (small)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['emotions_list'])
    print(f"Classes found: {mlb.classes_}")
    
    # 2. Extract Features (large)
    # We use np.stack which creates a copy. 
    # To save memory, we can try to force it to release the dataframe memory immediately.
    X = np.stack(df['embedding_vec'].values)
    
    # 3. Aggressively Free Memory
    print("Freeing DataFrame memory...")
    del df
    import gc
    gc.collect()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Free X and y since we have splits (copies)
    del X
    del y
    gc.collect()
    
    print(f"Training Data Shape: {X_train.shape}")
    print("Training MultiOutput Logistic Regression...")
    
    # Using SGDClassifier is much more memory efficient than lbfgs LogisticRegression
    # 'log_loss' gives probabilistic output (logistic regression)
    from sklearn.linear_model import SGDClassifier
    
    # SGD is much lighter on memory
    # Increased max_iter for better convergence
    clf = MultiOutputClassifier(SGDClassifier(loss='log_loss', max_iter=2000, random_state=42, n_jobs=-1))
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    print("Accuracy (Subset accuracy):", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))
    
    # Save model and binarizer
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    with open(BINARIZER_PATH, 'wb') as f:
        pickle.dump(mlb, f)
    print(f"Model saved to {MODEL_PATH}")
    
    return clf, mlb

def process_safety_data_in_batches(clf, mlb, batch_size=2000, limit=None):
    """Fetches IDs and embeddings from Safety table in batches, predicts, and updates."""
    print(f"Processing data from {SAFETY_TABLE} in batches...")
    
    # Check total count first (approximate)
    count_query = supabase.table(SAFETY_TABLE).select("id", count="exact").head()
    try:
        # For supabase-py < 2.0 (or async), count might be accessed differently
        # But 'head()' is not a method on SelectBuilder in newer versions
        # We just want to execute with count
        res = count_query.execute() 
        # Actually .head() is likely what caused the error if it doesn't exist.
        # Let's remove .head() and just limit 1
        count_query = supabase.table(SAFETY_TABLE).select("id", count="exact").limit(1)
        res = count_query.execute()
        total_rows = res.count
        print(f"Total rows to process: {total_rows}")
    except Exception as e:
        print(f"Could not get total count: {e}. Proceeding blindly...")
        total_rows = None

    offset = 0
    processed_count = 0
    
    while True:
        try:
            # Fetch a batch of embeddings
            print(f"Fetching batch (offset={offset})...")
            # We select ID and embedding. 
            r = supabase.table(SAFETY_TABLE).select("id, embedding").range(offset, offset + batch_size - 1).execute()
            data = r.data
            
            if not data:
                break
                
            batch_df = pd.DataFrame(data)
            batch_df['embedding_vec'] = batch_df['embedding'].apply(process_embedding_str)
            batch_df = batch_df.dropna(subset=['embedding_vec'])
            
            if not batch_df.empty:
                # Predict
                X_batch = np.stack(batch_df['embedding_vec'].values)
                y_pred = clf.predict(X_batch)
                y_labels = mlb.inverse_transform(y_pred)
                
                # Update
                updates = []
                for idx, row in batch_df.iterrows():
                    predicted_emotions = list(y_labels[idx])
                    updates.append({
                        "id": row['id'],
                        "predicted_emotions": predicted_emotions
                    })
                
                # Upload updates 1-by-1 (Supabase limitation for unrelated rows) or try batching if possible
                # Parallelize the updates?
                # For safety, let's just do sequential for now but suppress progress bar spam
                print(f"  > Uploading predictions for {len(updates)} rows...")
                for update in updates:
                    try:
                        supabase.table(SAFETY_TABLE).update({
                            "predicted_emotions": update["predicted_emotions"]
                        }).eq("id", update["id"]).execute()
                    except Exception as e:
                        print(f"    Failed update ID {update['id']}: {e}")
            
            offset += len(data)
            processed_count += len(data)
            
            # GC explicit
            import gc
            del data
            del batch_df
            gc.collect()
            
            if limit and processed_count >= limit:
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
        # Load model if not trained in this run
        if clf is None:
            if os.path.exists(MODEL_PATH) and os.path.exists(BINARIZER_PATH):
                print("Loading saved model...")
                with open(MODEL_PATH, 'rb') as f:
                    clf = pickle.load(f)
                with open(BINARIZER_PATH, 'rb') as f:
                    mlb = pickle.load(f)
            else:
                print("Model not found. Please run with --train first.")
                return
        
        # Stream processing instead of loading all
        process_safety_data_in_batches(clf, mlb, batch_size=1000, limit=args.limit)

if __name__ == "__main__":
    main()

