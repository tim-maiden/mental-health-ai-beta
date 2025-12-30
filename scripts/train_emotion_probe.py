import os
import sys
import json
import ast
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
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
    
    # --- SUPER-CLASS TRAINING (Positive / Negative) ---
    print("Mapping 28 Emotions to 2 Binaries (Positive, Negative)...")
    print("Note: 'Neutral' maps to [0,0] and 'Ambiguous' maps to [1,1]")
    
    # Define the Buckets
    POS_BUCKET = {"amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"}
    NEG_BUCKET = {"fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"}
    # Ambiguous emotions (high arousal, unclear valence) map to BOTH positive and negative
    AMB_BUCKET = {"realization", "surprise", "curiosity", "confusion"}
    # 'neutral' is excluded from all buckets -> [0, 0]

    def map_to_superclass(emotions_list):
        # We want to determine if 'positive' and 'negative' apply
        super_classes = set()
        
        emo_set = set(emotions_list)
        
        # Check Positive
        # Is positive if it has a positive emotion OR an ambiguous emotion
        if not emo_set.isdisjoint(POS_BUCKET) or not emo_set.isdisjoint(AMB_BUCKET):
            super_classes.add("positive")
        
        # Check Negative
        # Is negative if it has a negative emotion OR an ambiguous emotion
        if not emo_set.isdisjoint(NEG_BUCKET) or not emo_set.isdisjoint(AMB_BUCKET):
            super_classes.add("negative")
            
        return list(super_classes)
        
    df['emotions_list'] = df['emotions_list'].apply(map_to_superclass)
    
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
    
    if len(mlb.classes_) == 0:
        print("Error: No classes found in training data!")
        return None, None

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

    # --- IMPROVEMENT: Standardization ---
    from sklearn.preprocessing import StandardScaler
    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training Data Shape: {X_train.shape}")
    print("Training MultiOutput Logistic Regression (SGD) with Hyperparameter Optimization...")
    
    # Using SGDClassifier is much more memory efficient than lbfgs LogisticRegression
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import RandomizedSearchCV
    
    # Define base estimator
    sgd = SGDClassifier(
        loss='log_loss', 
        max_iter=2000, 
        random_state=42, 
        n_jobs=-1, 
        class_weight='balanced',
        early_stopping=True
    )
    
    # Simple parameter grid for optimization
    param_dist = {
        'estimator__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'estimator__penalty': ['l2', 'l1', 'elasticnet'],
    }
    
    # MultiOutputClassifier parameter search requires 'estimator__' prefix
    clf = MultiOutputClassifier(sgd)
    
    # We use a small subset for hyperparameter tuning to be fast
    print("Tuning hyperparameters on subset...")
    # Use 10% of training data for tuning to save time
    tune_size = min(20000, len(X_train))
    indices = np.random.choice(len(X_train), tune_size, replace=False)
    X_tune = X_train[indices]
    y_tune = y_train[indices]

    search = RandomizedSearchCV(
        clf, 
        param_distributions=param_dist, 
        n_iter=10, 
        scoring='f1_weighted', 
        n_jobs=-1, 
        cv=3,
        random_state=42,
        verbose=1
    )
    search.fit(X_tune, y_tune)
    
    print(f"Best params: {search.best_params_}")
    best_clf = search.best_estimator_
    
    # Retrain on full dataset
    print("Retraining best model on full dataset...")
    best_clf.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    # Get probabilities
    # For MultiOutputClassifier with SGD(loss='log_loss'), predict_proba returns list of arrays
    y_probs_list = best_clf.predict_proba(X_test)
    
    # Stack to get (n_samples, n_classes) matrix of probability for class=1
    # Each element in y_probs_list is (n_samples, 2), we want column 1.
    y_probs = np.array([class_probs[:, 1] for class_probs in y_probs_list]).T
    
    print(f"Probabilities shape: {y_probs.shape}")
    
    # --- THRESHOLD CALIBRATION ---
    print("Calibrating thresholds for max F1...")
    best_thresholds = []
    
    # We iterate through each class (column)
    for i, class_label in enumerate(mlb.classes_):
        y_true = y_test[:, i]
        y_score = y_probs[:, i]
        
        best_f1 = 0
        best_th = 0.5
        
        # Test thresholds from 0.1 to 0.9
        for th in np.arange(0.1, 1.0, 0.05):
            y_pred_th = (y_score >= th).astype(int)
            score = f1_score(y_true, y_pred_th)
            
            if score > best_f1:
                best_f1 = score
                best_th = th
                
        best_thresholds.append(best_th)
        print(f"  Class '{class_label}': Best Threshold={best_th:.2f} -> F1={best_f1:.4f}")
        
    print(f"Using Calibrated Thresholds: {best_thresholds}")
    
    # Apply new thresholds
    y_pred_calibrated = np.zeros_like(y_probs)
    for i in range(len(mlb.classes_)):
        y_pred_calibrated[:, i] = (y_probs[:, i] >= best_thresholds[i]).astype(int)
        
    print("Accuracy (Subset accuracy) [Calibrated]:", accuracy_score(y_test, y_pred_calibrated))
    print(classification_report(y_test, y_pred_calibrated, target_names=mlb.classes_, zero_division=0))
    
    # Save model, binarizer, AND scaler, AND thresholds
    # We'll save thresholds as a simple dictionary or list alongside the model
    THRESHOLDS_PATH = os.path.join(MODELS_DIR, "emotion_thresholds.json")
    
    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_clf, f)
    with open(BINARIZER_PATH, 'wb') as f:
        pickle.dump(mlb, f)
    SCALER_PATH = os.path.join(MODELS_DIR, "emotion_scaler.pkl")
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    with open(THRESHOLDS_PATH, 'w') as f:
        json.dump(best_thresholds, f)
        
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
    print(f"Thresholds saved to {THRESHOLDS_PATH}")
    
    return best_clf, mlb

def process_safety_data_in_batches(clf, mlb, batch_size=2000, limit=None, scaler=None, thresholds=None):
    """Fetches IDs and embeddings from Safety table in batches, predicts, and updates."""
    print(f"Processing data from {SAFETY_TABLE} in batches...")
    
    # Check total count first (approximate)
    count_query = supabase.table(SAFETY_TABLE).select("id", count="exact").limit(1)
    try:
        res = count_query.execute()
        total_rows = res.count
        print(f"Total rows to process: {total_rows}")
    except Exception as e:
        print(f"Could not get total count: {e}. Proceeding blindly...")
        total_rows = None

    offset = 0
    processed_count = 0
    
    # --- Prediction Configuration ---
    # Custom threshold to avoid "zero-label" issue with MultiOutputClassifier.predict()
    THRESHOLD = 0.5 
    
    from concurrent.futures import ThreadPoolExecutor

    def push_update(item):
        try:
            supabase.table(SAFETY_TABLE).update({
                "predicted_emotions": item["predicted_emotions"]
            }).eq("id", item["id"]).execute()
            return True
        except Exception as e:
            # print(f"    Failed update ID {item['id']}: {e}") # Reduce spam
            return False

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
                
                # Apply scaler if present
                if scaler:
                    X_batch = scaler.transform(X_batch)
                
                # Use predict_proba + Custom Threshold
                # MultiOutputClassifier with SGD returns a list of arrays (one per class)
                probas_list = clf.predict_proba(X_batch)
                
                # Stack to get (n_samples, n_classes) matrix of probability for class=1
                probas = np.array([class_probs[:, 1] for class_probs in probas_list]).T
                
                # Apply custom thresholds if available
                if thresholds:
                    # Broadcast thresholds across samples
                    y_pred_bool = probas > np.array(thresholds)
                else:
                    y_pred_bool = probas > THRESHOLD
                
                # Note: We NO LONGER force a label. [0, 0] corresponds to Neutral.
                
                # Inverse transform to get labels
                y_labels = mlb.inverse_transform(y_pred_bool)
                
                # Update List
                updates = []
                for idx, row in batch_df.iterrows():
                    predicted_emotions = list(y_labels[idx])
                    updates.append({
                        "id": row['id'],
                        "predicted_emotions": predicted_emotions
                    })
                
                # Parallelize the updates
                print(f"  > Uploading predictions for {len(updates)} rows (Parallelized)...")
                with ThreadPoolExecutor(max_workers=10) as executor:
                    list(executor.map(push_update, updates))
            
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
            # We just trained, so we don't have thresholds variable in this scope easily unless we returned it
            # For now, let's assume default 0.5 if running immediately after train, OR reload them
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
