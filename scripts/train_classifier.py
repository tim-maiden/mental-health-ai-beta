import numpy as np
import os
import sys
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, set_seed

# Set seed for reproducibility
set_seed(42)

# Disable tokenizer parallelism to prevent deadlocks with DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.modeling.training import get_training_args, compute_metrics, CustomTrainer
from src.config import (
    MODEL_ID, 
    TRAIN_FILE, 
    TEST_FILE, 
    MODEL_OUTPUT_DIR, 
    WANDB_PROJECT,
    DATA_DIR,
    TRAIN_EPOCHS
)

OUTPUT_DIR = MODEL_OUTPUT_DIR
NUM_EPOCHS = TRAIN_EPOCHS

def main():
    print(f"--- Loading Data from Local Files ---")
    
    # Authenticate using environment variable explicitly
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        
    # Attempt online initialization with timeout; fallback to offline mode on failure.
    # Log masked API key for debugging.
    try:
        print(f"--- Initializing WandB (Project: {WANDB_PROJECT}) ---")
        # Mask the key for logging safety (show first 4 chars only if present)
        masked_key = os.environ.get("WANDB_API_KEY", "")
        if masked_key:
            masked_key = masked_key[:4] + "*" * (len(masked_key) - 4)
            print(f"WANDB_API_KEY detected: {masked_key}")
        else:
            print("WANDB_API_KEY is NOT set in environment variables!")

        # Explicitly set 300s timeout to match RunPod environment realities
        print("Attempting W&B initialization (Timeout: 300s)...")
        wandb.init(project=WANDB_PROJECT, settings=wandb.Settings(init_timeout=300))
    except Exception as e:
        print(f"\nWarning: WandB online initialization failed: {e}")
        print("Falling back to offline mode. Run `wandb sync` later to upload logs.")
        wandb.init(project=WANDB_PROJECT, mode="offline")

    dataset = load_dataset("parquet", data_files={
        "train": TRAIN_FILE,
        "test": TEST_FILE
    })
    
    # Load subreddit mapping to determine num_labels
    mapping_path = os.path.join(DATA_DIR, "subreddit_mapping.json")
    with open(mapping_path, "r") as f:
        subreddit_map = json.load(f)
    num_labels = len(subreddit_map)
    print(f"Loaded {num_labels} classes from subreddit_mapping.json")

    print(f"--- Loading Tokenizer ({MODEL_ID}) ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    def preprocess_function(examples):
        # Tokenize
        tokenized = tokenizer(examples["text"], truncation=True, max_length=512)
        # Map 'soft_label' to 'labels' key to ensure CustomTrainer computes loss against soft targets.
        tokenized["labels"] = examples["soft_label"]
        return tokenized

    # Remove raw text and original label columns to prevent DataCollator conflicts.
    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset["train"].column_names # Remove all original cols to be safe
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"--- Initializing Model ---")
    # For soft targets, we still use SequenceClassification, but we ignore the default loss in our Trainer
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, 
        num_labels=num_labels,
        problem_type="multi_label_classification" # Explicitly set problem_type for compatibility, though CustomTrainer overrides the loss function.
    )

    # Full fine-tuning (all layers trainable) to optimize for domain shift.
    print("Model is FULLY TRAINABLE (No Freezing).")

    # Load Risk Indices
    risk_indices_path = os.path.join(DATA_DIR, "risk_indices.json")
    risk_indices = None
    if os.path.exists(risk_indices_path):
        with open(risk_indices_path, "r") as f:
            risk_indices = json.load(f)
        print(f"Loaded {len(risk_indices)} risk indices for Hierarchical Loss.")
    else:
        print("Warning: risk_indices.json not found. Hierarchical Loss will be disabled.")

    # Note: Model compilation is now handled by Trainer via torch_compile parameter
    # This avoids signature inspection issues with torch.compile()

    training_args = get_training_args(
        output_dir=OUTPUT_DIR,
        model_id=MODEL_ID,
        train_size=len(dataset['train']),
        num_epochs=NUM_EPOCHS
    )
    
    # H100 / Blackwell Optimization Strategy
    # Maximize VRAM usage (H100 has 80GB, RTX 6000 Blackwell has 96GB)
    if os.getenv("DEPLOY_ENV") in ["runpod", "cloud"]:
        print("Overriding Training Args for High-VRAM Optimization (H100/Blackwell)...")
        training_args.per_device_train_batch_size = 64
        training_args.gradient_accumulation_steps = 1 # Global batch size = 64 (per device) * 1 * N_devices
        training_args.bf16 = True
        training_args.fp16 = False # Ensure FP16 is off
        training_args.dataloader_num_workers = 8
        
    # Ensure correct metric is used despite caching issues
    training_args.metric_for_best_model = "eval_accuracy"

    trainer = CustomTrainer(
        risk_indices=risk_indices,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("--- Starting Training (Multilabel/Soft-Label) ---")
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    trainer.train()

    print(f"--- Saving Model to {OUTPUT_DIR} ---")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Copy subreddit_mapping.json to output dir so it gets uploaded with the model
    try:
        import shutil
        mapping_src = os.path.join(DATA_DIR, "subreddit_mapping.json")
        mapping_dst = os.path.join(OUTPUT_DIR, "subreddit_mapping.json")
        if os.path.exists(mapping_src):
            shutil.copy(mapping_src, mapping_dst)
            print(f"Copied subreddit_mapping.json to {OUTPUT_DIR}")
            
        # [NEW] Copy risk_indices.json as well (for provenance/reproducibility)
        indices_src = os.path.join(DATA_DIR, "risk_indices.json")
        indices_dst = os.path.join(OUTPUT_DIR, "risk_indices.json")
        if os.path.exists(indices_src):
            shutil.copy(indices_src, indices_dst)
            print(f"Copied risk_indices.json to {OUTPUT_DIR}")
            
    except Exception as e:
        print(f"Warning: Failed to copy auxiliary JSON files: {e}")

    # [NEW] Calculate Optimal Threshold
    if risk_indices:
        print("\n--- Calculating Optimal Risk Threshold ---")
        try:
            # Get predictions on validation set
            predictions = trainer.predict(tokenized_datasets["test"])
            logits = torch.tensor(predictions.predictions)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Calculate Risk Probability Mass (Sum of all risk class probs)
            risk_probs = torch.sum(probs[:, risk_indices], dim=1).numpy()
            
            # True Labels (Binary)
            # We need to map the soft labels back to binary risk/safe for this check
            # Or just use the original test dataframe if available.
            # Since we only have tokenized_datasets["test"], we can infer risk from 'labels' (soft targets)
            test_soft_labels = tokenized_datasets["test"]["labels"]
            test_soft_labels = torch.tensor(test_soft_labels)
            
            # If sum of risk class probs in TARGET > 0.5, it's a risk sample
            true_risk_mass = torch.sum(test_soft_labels[:, risk_indices], dim=1).numpy()
            true_binary = (true_risk_mass > 0.5).astype(int)
            
            from sklearn.metrics import f1_score
            best_f1 = 0
            best_thresh = 0.5
            
            # Simple grid search
            for thresh in np.arange(0.1, 0.95, 0.05):
                pred_binary = (risk_probs > thresh).astype(int)
                f1 = f1_score(true_binary, pred_binary, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            
            print(f"Best Threshold: {best_thresh:.2f} (F1: {best_f1:.3f})")
            
            # Save threshold.json
            thresh_data = {"risk_threshold": float(best_thresh), "f1_score": float(best_f1)}
            with open(os.path.join(OUTPUT_DIR, "threshold.json"), "w") as f:
                json.dump(thresh_data, f)
            print(f"Saved threshold.json to {OUTPUT_DIR}")
            
        except Exception as e:
            print(f"Warning: Failed to calculate threshold: {e}")
            import traceback
            traceback.print_exc()

    # Push to Hub if configured
    try:
        print("Pushing model to Hugging Face Hub...")
        # This will push the model to your-username/risk_classifier_deberta_small_v1 (or large)
        # It relies on you being logged in or having HF_TOKEN env var set.
        repo_name = f"risk-classifier-deberta-{MODEL_ID.split('/')[-1]}"
        trainer.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
        print(f"Successfully pushed model to {repo_name}")
    except Exception as e:
        print(f"Warning: Failed to push model to Hub: {e}")
    
    print("\n--- Final Evaluation ---")
    print("Evaluating on Test Set...")
    # Skip standard evaluation during training to reduce overhead.
    # Evaluation is handled in detail by scripts/inference.py and WandB logging.
    # metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"], metric_key_prefix="eval")
    # print(metrics)
    
    wandb.finish()

if __name__ == "__main__":
    main()
