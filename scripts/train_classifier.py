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
    DATA_DIR
)

OUTPUT_DIR = MODEL_OUTPUT_DIR
NUM_EPOCHS = 2 # Reduced from 10 to 2 for Abundance Modeling (2.4M rows)

def main():
    print(f"--- Loading Data from Local Files ---")
    
    # Authenticate using environment variable explicitly
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        
    # Robust Initialization: Try online, fallback to offline if timeout
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

    # 1. Update paths to look for .parquet
    train_path = TRAIN_FILE.replace('.jsonl', '.parquet')
    test_path = TEST_FILE.replace('.jsonl', '.parquet')

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Did you run compile_dataset.py?")
        sys.exit(1)

    print(f"Loading Parquet datasets from {train_path}...")

    # 2. Load using the "parquet" builder
    dataset = load_dataset("parquet", data_files={
        "train": train_path,
        "test": test_path
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
        # CRITICAL: Map 'soft_label' to 'labels' for the CustomTrainer
        tokenized["labels"] = examples["soft_label"]
        return tokenized

    # CRITICAL FIX: Remove the raw 'text' and old 'label' columns so Trainer doesn't get confused
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
        problem_type="multi_label_classification" # Hint to HF (though we override loss anyway)
    )

    # --- UNFREEZING: FULLY TRAINABLE ---
    # We removed the freezing logic. The model is now fully trainable to overwrite pre-trained biases.
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
    
    # FORCE OVERRIDE: H100 Optimization Strategy
    # Maximize VRAM usage (H100 has 80GB)
    if os.getenv("DEPLOY_ENV") in ["runpod", "cloud"]:
        print("Overriding Training Args for H100 Abundance Strategy...")
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
    
    print("\n--- Final Evaluation ---")
    print("Evaluating on Test Set...")
    # NOTE: We skip standard evaluation during training script to avoid overhead.
    # Evaluation is handled in detail by scripts/inference.py and WandB logging.
    # metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"], metric_key_prefix="eval")
    # print(metrics)
    
    wandb.finish()

if __name__ == "__main__":
    main()
