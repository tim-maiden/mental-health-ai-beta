import os
import sys
import argparse
import pandas as pd
import json
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.modeling.inference import load_model, predict_batch, is_clean_english, get_device
from src.core.clients import supabase  # Import supabase client
from datasets import load_dataset, Dataset

# Configuration - Use environment-aware paths from config
from src.config import MODEL_OUTPUT_DIR, DATA_DIR, OUTPUT_DIR

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
    """
    Deprecated: Prefer loading WildChat from Supabase or HF.
    Loads lmsys-chat-1m dataset in stream mode.
    """
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

def load_wildchat_generator_from_hf(batch_size=32, limit=None):
    """
    Generator that fetches WildChat chunks from Hugging Face directly (Streaming).
    """
    print(f"Loading allenai/WildChat-1M dataset from Hugging Face (Stream mode)...")
    dataset = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    
    batch_texts = []
    batch_ids = []
    count = 0
    
    for row in dataset:
        conversation = row['conversation']
        row_id = row['conversation_hash'] # WildChat uses hash as ID
        
        # Iterate through turns to find user messages
        # WildChat structure: list of dicts with 'role' and 'content'
        for i, turn in enumerate(conversation):
            if turn['role'] == 'user':
                user_text = turn['content']
                
                # Basic filtering
                if user_text and isinstance(user_text, str) and len(user_text.strip()) > 5:
                    batch_texts.append(user_text)
                    batch_ids.append(f"{row_id}_{i}")
                    
                    if len(batch_texts) == batch_size:
                        yield batch_ids, batch_texts
                        batch_texts = []
                        batch_ids = []
                    
                    count += 1
                    if limit and count >= limit:
                        break
        if limit and count >= limit:
            break
            
    if batch_texts:
        yield batch_ids, batch_texts

def load_wildchat_generator_from_supabase(batch_size=32, limit=None):
    """
    Generator that fetches WildChat chunks from Supabase.
    """
    print(f"Loading WildChat data from Supabase (Stream mode)...")
    
    offset = 0
    total_fetched = 0
    fetch_size = 1000  # Number of rows to fetch from DB at once
    
    # Buffer to hold rows from DB until we yield them in batch_size chunks
    buffer_ids = []
    buffer_texts = []
    
    while True:
        # Check if we hit the limit
        if limit and total_fetched >= limit:
            break

        # Fetch batch from Supabase
        try:
            # Adjust fetch_size if we're near the limit
            current_fetch = fetch_size
            if limit and (limit - total_fetched) < fetch_size:
                current_fetch = limit - total_fetched
                
            response = supabase.table("wildchat_embeddings")\
                .select("id, input")\
                .range(offset, offset + current_fetch - 1)\
                .execute()
            
            rows = response.data
            if not rows:
                break # No more data
                
            offset += len(rows)
            total_fetched += len(rows)
            
            # Add to buffer
            for row in rows:
                buffer_ids.append(row['id'])
                buffer_texts.append(row['input'])
                
                # Yield when buffer is full enough
                if len(buffer_ids) >= batch_size:
                    yield buffer_ids[:batch_size], buffer_texts[:batch_size]
                    # Remove yielded items
                    buffer_ids = buffer_ids[batch_size:]
                    buffer_texts = buffer_texts[batch_size:]
            
        except Exception as e:
            print(f"Error fetching from Supabase: {e}")
            break
            
    # Yield remaining
    if buffer_ids:
        yield buffer_ids, buffer_texts

def run_inference(args):
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        model_path = STANDARD_PATH
        if not os.path.exists(STANDARD_PATH):
            # Fallback: try legacy path (for backward compatibility)
            legacy_path = "models/risk_classifier_deberta_v1"
            if os.path.exists(legacy_path):
                model_path = legacy_path
            else:
                # Keep original error if local default not found and no arg provided
                # But allow HF loading if user provides it via args.model
                pass 

    print(f"Loading Model from: {model_path}")
    
    kwargs = {}
    if args.subfolder:
        kwargs['subfolder'] = args.subfolder

    model, tokenizer, device = load_model(model_path, **kwargs)
        
    if args.wildchat:
        default_output = os.path.join(OUTPUT_DIR, "wildchat_silver_labels.pkl")
        # Use HF loader if requested, otherwise fallback to Supabase (or make HF default if desired)
        # Given user preference for HF:
        if args.source == "supabase":
             data_gen = load_wildchat_generator_from_supabase(batch_size=args.batch_size, limit=args.limit)
             dataset_name = "WildChat (Supabase)"
        else:
             data_gen = load_wildchat_generator_from_hf(batch_size=args.batch_size, limit=args.limit)
             dataset_name = "WildChat (HF)"

    else:
        default_output = os.path.join(OUTPUT_DIR, "lmsys_risk_scores.pkl")
        data_gen = load_lmsys_generator(batch_size=args.batch_size, limit=args.limit)
        dataset_name = "LMSYS (HF)"

    output_file = args.output if args.output else default_output
    print(f"Collecting results to save to: {output_file}")
    
    all_results = []
    
    # Use limit for TQDM total if provided, otherwise default to infinite counter.
    pbar = tqdm(data_gen, total=(args.limit // args.batch_size) if args.limit else None, unit="batch", desc=f"Processing {dataset_name}")
    
    for batch_ids, batch_texts in pbar:
        keep_indices = []
        skip_indices = []
        
        for i, text in enumerate(batch_texts):
            # Basic validation
            if text and isinstance(text, str) and len(text.strip()) > 0:
                 # Validate input length (language filtering is handled during data loading).
                 keep_indices.append(i)
            else:
                skip_indices.append(i)
        
        if keep_indices:
            valid_texts = [batch_texts[i] for i in keep_indices]
            probs = predict_batch(model, tokenizer, valid_texts, device)
            
            # Load mapping
            try:
                mapping_path = os.path.join(DATA_DIR, "subreddit_mapping.json")
                with open(mapping_path, "r") as f:
                    mapping = json.load(f)
                id_to_sub = {v: k for k, v in mapping.items()}
            except:
                # print("Warning: Could not load subreddit mapping. Using raw indices.")
                id_to_sub = {}

            for idx, prob in zip(keep_indices, probs):
                # prob is now a distribution over N subreddits
                # Find top class
                top_idx = torch.argmax(prob).item()
                top_prob = prob[top_idx].item()
                top_label = id_to_sub.get(top_idx, str(top_idx))
                
                # Output top label and confidence score.
                
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
    
    # Hugging Face Upload
    if args.upload_dataset:
        print(f"--- Uploading Silver Labels to HF: {args.dataset_id} ---")
        try:
            # Filter out rows where full_dist is empty (failed predictions)
            # This ensures the training dataset only contains valid probability distributions
            df_upload = df_results[df_results['full_dist'].map(len) > 0].copy()
            print(f"Filtered {len(df_results) - len(df_upload)} invalid rows. Uploading {len(df_upload)} valid samples.")
            
            # Convert filtered DataFrame to HF Dataset
            # Ensure columns are friendly types (lists are fine)
            hf_dataset = Dataset.from_pandas(df_upload)
            
            # Push to Hub (Private by default)
            hf_dataset.push_to_hub(args.dataset_id, private=True)
            print(f"Successfully uploaded dataset to https://huggingface.co/datasets/{args.dataset_id}")
        except Exception as e:
            print(f"Error uploading to Hugging Face: {e}")

    print(f"Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on test texts or LMSYS/WildChat dataset.")
    parser.add_argument("--lmsys", action="store_true", help="Run inference on LMSYS chat dataset")
    parser.add_argument("--wildchat", action="store_true", help="Run inference on WildChat dataset")
    parser.add_argument("--source", type=str, default="hf", choices=["hf", "supabase"], help="Source for WildChat data (default: hf)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--output", type=str, default=None, help="Output PKL file path")
    parser.add_argument("--model", type=str, default=None, help="Path to model or HF Repo ID")
    parser.add_argument("--subfolder", type=str, default=None, help="Subfolder in HF Repo")
    parser.add_argument("--upload-dataset", action="store_true", help="Upload the results as a HF Dataset")
    parser.add_argument("--dataset-id", type=str, default="tim-maiden/mental-health-silver-labels", help="Target HF Dataset ID")
    
    args = parser.parse_args()
    
    if args.lmsys or args.wildchat:
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
        else:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        run_inference(args)
    else:
        print("Running Standard Test Set...")
        if args.model:
            model_path = args.model
        else:
            model_path = STANDARD_PATH
            if not os.path.exists(STANDARD_PATH):
                # Fallback: try legacy path (for backward compatibility)
                legacy_path = "models/risk_classifier_deberta_v1"
                if os.path.exists(legacy_path):
                    model_path = legacy_path
                else:
                    # If local not found and not explicit, maybe raise error or let load_model fail
                    pass

        kwargs = {}
        if args.subfolder:
            kwargs['subfolder'] = args.subfolder

        model, tokenizer, device = load_model(model_path, **kwargs)
        probs = predict_batch(model, tokenizer, INPUT_TEXTS, device)
        
        # Load mapping
        id_to_sub = {}
        mapping_found = False
        
        # 1. Try loading from model directory (Portable)
        possible_paths = [
            os.path.join(model_path, "subreddit_mapping.json"),
            os.path.join(DATA_DIR, "subreddit_mapping.json")
        ]
        
        # If running in Cloud/RunPod, explicit check for /workspace/models
        if os.getenv("DEPLOY_ENV") in ["runpod", "cloud"]:
             possible_paths.append("/workspace/models/risk_classifier_deberta_large_v1/subreddit_mapping.json")

        # Try to download from HF if model arg looks like a repo ID
        if args.model and "/" in args.model:
            try:
                # print(f"Attempting to download subreddit_mapping.json from HF: {args.model}")
                subfolder = args.subfolder if args.subfolder else None
                downloaded_path = hf_hub_download(repo_id=args.model, filename="subreddit_mapping.json", subfolder=subfolder)
                possible_paths.append(downloaded_path)
            except Exception as e:
                # print(f"Could not download mapping from HF: {e}")
                pass

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        mapping = json.load(f)
                    id_to_sub = {v: k for k, v in mapping.items()}
                    # print(f"Loaded subreddit mapping from {path}")
                    mapping_found = True
                    break
                except Exception as e:
                    print(f"Warning: Failed to load mapping from {path}: {e}")
        
        if not mapping_found:
             print("Warning: Could not load subreddit mapping. Output will show raw indices.")

        # Load Risk Indices
        risk_indices = []
        possible_risk_paths = [
            os.path.join(model_path, "risk_indices.json"),
            os.path.join(DATA_DIR, "risk_indices.json")
        ]
         # Try to download from HF if needed
        if args.model and "/" in args.model:
            try:
                subfolder = args.subfolder if args.subfolder else None
                downloaded_risk_path = hf_hub_download(repo_id=args.model, filename="risk_indices.json", subfolder=subfolder)
                possible_risk_paths.append(downloaded_risk_path)
            except:
                pass

        for path in possible_risk_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        risk_indices = json.load(f)
                    print(f"Loaded {len(risk_indices)} risk indices for Risk Mass calculation.")
                    break
                except Exception as e:
                    print(f"Warning: Failed to load risk indices from {path}: {e}")

        print(f"\n{'TEXT':<60} | {'TOP LABEL':<20} | {'CONF'} | {'RISK MASS'}")
        print("-" * 105)
        
        results = []
        for text, prob in zip(INPUT_TEXTS, probs):
            top_idx = torch.argmax(prob).item()
            top_prob = prob[top_idx].item()
            label = id_to_sub.get(top_idx, str(top_idx))
            
            # Calculate Risk Mass
            risk_mass = 0.0
            if risk_indices:
                risk_mass = torch.sum(prob[risk_indices]).item()
            
            # Ambiguity Check (Fix #2)
            # If top confidence is very low, it might be OOD
            display_label = label
            if top_prob < 0.25:
                display_label = f"{label} (?)"
            
            print(f"{text[:58]:<60} | {display_label:<20} | {top_prob:.1%} | {risk_mass:.1%}")
            
            results.append({
                'text': text,
                'top_label': label,
                'confidence': top_prob,
                'risk_mass': risk_mass,
                'full_dist': prob.tolist()
            })
            
        if args.output:
             print(f"\nSaving results to {args.output}...")
             os.makedirs(os.path.dirname(args.output), exist_ok=True)
             pd.DataFrame(results).to_pickle(args.output)
             print("Done.")
