import os
import sys
import argparse
import pandas as pd
import json
import torch
from tqdm import tqdm
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.modeling.inference import load_model, predict_batch, is_clean_english, get_device
from datasets import load_dataset

# Configuration - Use environment-aware paths from config
from src.config import MODEL_OUTPUT_DIR, QUANTIZED_MODEL_DIR, DATA_DIR, OUTPUT_DIR

QUANTIZED_PATH = QUANTIZED_MODEL_DIR
STANDARD_PATH = MODEL_OUTPUT_DIR

# Load dynamic threshold if available
RISK_THRESHOLD = 0.5
threshold_file = os.path.join(STANDARD_PATH, "threshold.json")
if os.path.exists(threshold_file):
    try:
        with open(threshold_file, "r") as f:
            config = json.load(f)
            RISK_THRESHOLD = config.get("risk_threshold", 0.5)
            print(f"Loaded dynamic threshold: {RISK_THRESHOLD}")
    except Exception as e:
        print(f"Warning: Could not load threshold.json: {e}")

# Test Examples
INPUT_TEXTS = [
    "I'm feeling really great today, just beat my high score in the game!",
    "I'm smelly!",  
    "I don't see the point in going on anymore. I'm so tired of living.",    
    "I'm scared to talk to people at parties, I just stand in the corner.",   
    "My girlfriend dumped me and I feel so lonely right now.",
    "The CPU temperature is stable at 45 degrees celsius.",
    "Til that the moon is tidally locked to the earth.",
    "Can someone explain how a rotary engine works?",
    "The new graphics card update improves performance by 20%.",
    "I finally finished building my deck this weekend.",
    "I love hiking in the mountains, it clears my head."
]

def load_lmsys_generator(batch_size=32, limit=None):
    print(f"Loading lmsys-chat-1m dataset (Stream mode)...")
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    
    batch_texts = []
    batch_ids = []
    count = 0
    
    for row in dataset:
        conversation = row['conversation']
        row_id = row['conversation_id']
        
        user_text = None
        for turn in conversation:
            if turn['role'] == 'user':
                user_text = turn['content']
                break 
        
        if user_text:
            batch_texts.append(user_text)
            batch_ids.append(row_id)
            
            if len(batch_texts) == batch_size:
                yield batch_ids, batch_texts
                batch_texts = []
                batch_ids = []
                
            count += 1
            if limit and count >= limit:
                break
    
    if batch_texts:
        yield batch_ids, batch_texts

def load_wildchat_generator(batch_size=32, limit=None):
    print(f"Loading allenai/WildChat-1M dataset (Stream mode)...")
    dataset = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    
    batch_texts = []
    batch_ids = []
    count = 0
    
    for row in dataset:
        conversation = row['conversation']
        row_id = row['conversation_hash'] # Use conversation_hash
        if conversation is None: continue
        
        user_text = None
        for turn in conversation:
            if turn['role'] == 'user':
                user_text = turn['content']
                break 
        
        if user_text:
            batch_texts.append(user_text)
            batch_ids.append(row_id)
            
            if len(batch_texts) == batch_size:
                yield batch_ids, batch_texts
                batch_texts = []
                batch_ids = []
                
            count += 1
            if limit and count >= limit:
                break
    
    if batch_texts:
        yield batch_ids, batch_texts

def run_inference(args):
    # Determine model path
    is_quantized = False
    model_path = STANDARD_PATH
    if os.path.exists(os.path.join(QUANTIZED_PATH, "model_quantized.onnx")):
        model_path = QUANTIZED_PATH
        is_quantized = True
    elif not os.path.exists(STANDARD_PATH):
        # Fallback: try legacy path (for backward compatibility)
        legacy_path = "models/risk_classifier_deberta_v1"
        if os.path.exists(legacy_path):
            model_path = legacy_path
        else:
            raise FileNotFoundError(f"Model not found at {STANDARD_PATH} or {legacy_path}")

    print(f"Loading Model from: {model_path} (Quantized: {is_quantized})")
    
    model, tokenizer, device = load_model(model_path, is_quantized=is_quantized)
        
    if args.wildchat:
        default_output = os.path.join(OUTPUT_DIR, "wildchat_risk_scores.pkl")
        data_gen = load_wildchat_generator(batch_size=args.batch_size, limit=args.limit)
        dataset_name = "WildChat"
    else:
        default_output = os.path.join(OUTPUT_DIR, "lmsys_risk_scores.pkl")
        data_gen = load_lmsys_generator(batch_size=args.batch_size, limit=args.limit)
        dataset_name = "LMSYS"

    output_file = args.output if args.output else default_output
    print(f"Collecting results to save to: {output_file}")
    
    all_results = []
    
    pbar = tqdm(data_gen, total=(args.limit // args.batch_size) if args.limit else None, unit="batch", desc=f"Processing {dataset_name}")
    
    for batch_ids, batch_texts in pbar:
        keep_indices = []
        skip_indices = []
        
        for i, text in enumerate(batch_texts):
            if is_clean_english(text):
                keep_indices.append(i)
            else:
                skip_indices.append(i)
        
        # Prepare batch results container
        # We'll just append dicts to all_results
        
        if keep_indices:
            valid_texts = [batch_texts[i] for i in keep_indices]
            probs = predict_batch(model, tokenizer, valid_texts, device, is_quantized=is_quantized)
            
            # Load mapping
            try:
                mapping_path = os.path.join(DATA_DIR, "subreddit_mapping.json")
                with open(mapping_path, "r") as f:
                    mapping = json.load(f)
                id_to_sub = {v: k for k, v in mapping.items()}
            except:
                print("Warning: Could not load subreddit mapping. Using raw indices.")
                id_to_sub = {}

            for idx, prob in zip(keep_indices, probs):
                # prob is now a distribution over N subreddits
                # Find top class
                top_idx = torch.argmax(prob).item()
                top_prob = prob[top_idx].item()
                top_label = id_to_sub.get(top_idx, str(top_idx))
                
                # Heuristic Risk Score: Sum of probs of known risk subreddits?
                # Or just output the top prediction.
                # For now, let's output top label + prob.
                
                all_results.append({
                    'id': batch_ids[idx],
                    'text': batch_texts[idx],
                    'top_label': top_label,
                    'confidence': float(top_prob),
                    'full_dist': prob.tolist()
                })

        for idx in skip_indices:
            all_results.append({
                'id': batch_ids[idx],
                'text': batch_texts[idx],
                'top_label': "N/A",
                'confidence': 0.0,
                'full_dist': []
            })
            
    # Save to PKL
    print(f"Saving {len(all_results)} results to {output_file}...")
    df_results = pd.DataFrame(all_results)
    df_results.to_pickle(output_file)
    print(f"Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on test texts or LMSYS/WildChat dataset.")
    parser.add_argument("--lmsys", action="store_true", help="Run inference on LMSYS chat dataset")
    parser.add_argument("--wildchat", action="store_true", help="Run inference on WildChat dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--output", type=str, default=None, help="Output PKL file path")
    
    args = parser.parse_args()
    
    if args.lmsys or args.wildchat:
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
        else:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        run_inference(args)
    else:
        print("Running Standard Test Set...")
        is_quantized = False
        model_path = STANDARD_PATH
        if os.path.exists(os.path.join(QUANTIZED_PATH, "model_quantized.onnx")):
            model_path = QUANTIZED_PATH
            is_quantized = True
        elif not os.path.exists(STANDARD_PATH):
            # Fallback: try legacy path (for backward compatibility)
            legacy_path = "models/risk_classifier_deberta_v1"
            if os.path.exists(legacy_path):
                model_path = legacy_path
            else:
                raise FileNotFoundError(f"Model not found at {STANDARD_PATH} or {legacy_path}")

        model, tokenizer, device = load_model(model_path, is_quantized=is_quantized)
        probs = predict_batch(model, tokenizer, INPUT_TEXTS, device, is_quantized=is_quantized)
        
        # Load mapping
        try:
            mapping_path = os.path.join(DATA_DIR, "subreddit_mapping.json")
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
            id_to_sub = {v: k for k, v in mapping.items()}
        except:
            print("Warning: Could not load subreddit mapping.")
            id_to_sub = {}

        print(f"\n{'TEXT':<60} | {'TOP LABEL':<20} | {'CONF'}")
        print("-" * 90)
        for text, prob in zip(INPUT_TEXTS, probs):
            top_idx = torch.argmax(prob).item()
            top_prob = prob[top_idx].item()
            label = id_to_sub.get(top_idx, str(top_idx))
            
            print(f"{text[:58]:<60} | {label:<20} | {top_prob:.1%}")
