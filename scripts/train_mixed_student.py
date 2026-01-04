import os
import sys
import pandas as pd
import numpy as np
import torch
import wandb
import json
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.modeling.training import get_training_args, CustomTrainer, compute_metrics
from src.config import (
    MODEL_OUTPUT_DIR, 
    WANDB_PROJECT,
    DATA_DIR,
    STUDENT_MODEL_ID,
    DISTILLATION_EPOCHS,
    DISTILLATION_OUTPUT_DIR,
    DEPLOY_ENV
)

# Config
WILDCHAT_DATASET_ID = "tim-maiden/mental-health-silver-labels"
REDDIT_DATASET_ID = "tim-maiden/mental-health-ai"
OUTPUT_DIR = DISTILLATION_OUTPUT_DIR
NUM_EPOCHS = DISTILLATION_EPOCHS

def to_soft_label(hard_label, num_classes):
    """
    Converts a hard integer label to a one-hot soft label distribution.
    """
    distribution = [0.0] * num_classes
    if 0 <= hard_label < num_classes:
        distribution[hard_label] = 1.0
    return distribution

def main():
    print(f"--- Starting Mixed Student Distillation (Model: {STUDENT_MODEL_ID}) ---")
    print(f"--- Strategy: Mixed Training (WildChat Silver + Reddit Gold) ---")
    
    # 1. WandB Init
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])
    
    try:
        wandb.init(project=WANDB_PROJECT, name="student-distillation-mixed", settings=wandb.Settings(init_timeout=300))
    except:
        print("WandB init failed, using offline mode.")
        wandb.init(project=WANDB_PROJECT, mode="offline", name="student-distillation-mixed")

    # 2. Load WildChat Data (Silver)
    print(f"Loading WildChat Silver Labels from: {WILDCHAT_DATASET_ID}...")
    wildchat_ds = load_dataset(WILDCHAT_DATASET_ID, split="train")
    if "labels" not in wildchat_ds.column_names and "full_dist" in wildchat_ds.column_names:
        wildchat_ds = wildchat_ds.rename_column("full_dist", "labels")
    
    # Split WildChat
    wildchat_split = wildchat_ds.train_test_split(test_size=0.1, seed=42)
    wildchat_train = wildchat_split["train"]
    wildchat_test = wildchat_split["test"]
    
    # 3. Load Reddit Data (Gold)
    print(f"Loading Reddit Gold Data from: {REDDIT_DATASET_ID}...")
    # Attempt to load from HF. If it's private, ensure HF_TOKEN is set.
    reddit_ds = load_dataset(REDDIT_DATASET_ID, split="train") 
    
    # Determine num_labels from WildChat (which has the full distribution)
    example_labels = wildchat_train[0]['labels']
    num_labels = len(example_labels)
    print(f"Detected {num_labels} classes from Silver labels.")

    # Preprocess Reddit Data to match WildChat format
    # Reddit data has 'label' (int) and 'text' (str). We need 'labels' (list<float>).
    # Check if 'label' column exists, if not check 'binary_label' or similar
    
    # Note: The reddit parquet usually has 'label' as the target integer.
    # We need to map this to soft labels.
    
    def process_reddit(example):
        # Create soft label
        label_id = example.get('label')
        if label_id is None:
             # Fallback if label col is missing (should not happen if dataset is correct)
             return {"labels": [0.0] * num_labels}
        
        return {"labels": to_soft_label(label_id, num_labels)}

    print("Preprocessing Reddit data to generate soft labels...")
    # Filter out columns we don't need to avoid schema mismatches
    cols_to_keep = ['text', 'label']
    reddit_clean = reddit_ds.select_columns([c for c in cols_to_keep if c in reddit_ds.column_names])
    
    # Apply transformation
    reddit_mapped = reddit_clean.map(process_reddit)
    
    # Remove original 'label' column to match WildChat schema completely
    if 'label' in reddit_mapped.column_names:
        reddit_mapped = reddit_mapped.remove_columns(['label'])
    
    # Ensure text column match
    if 'input' in reddit_mapped.column_names and 'text' not in reddit_mapped.column_names:
         reddit_mapped = reddit_mapped.rename_column('input', 'text')

    # Split Reddit
    # Use a fixed seed to ensure we don't leak train data into test if we re-run
    reddit_split = reddit_mapped.train_test_split(test_size=0.1, seed=42)
    reddit_train = reddit_split["train"]
    reddit_test = reddit_split["test"]
    
    print(f"Data Stats:")
    print(f"  WildChat Train: {len(wildchat_train)}")
    print(f"  Reddit Train:   {len(reddit_train)}")
    print(f"  WildChat Test:  {len(wildchat_test)}")
    print(f"  Reddit Test:    {len(reddit_test)}")
    
    # 4. Concatenate Training Data
    combined_train = concatenate_datasets([wildchat_train, reddit_train])
    combined_train = combined_train.shuffle(seed=42) # Important to shuffle mixed data
    print(f"Combined Training Set: {len(combined_train)}")

    # 5. Tokenizer & Model
    print(f"Loading Tokenizer ({STUDENT_MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    print("Tokenizing datasets...")
    tokenized_train = combined_train.map(preprocess_function, batched=True)
    tokenized_wildchat_test = wildchat_test.map(preprocess_function, batched=True)
    tokenized_reddit_test = reddit_test.map(preprocess_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    print(f"Initializing Model ({STUDENT_MODEL_ID}) with {num_labels} labels...")
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL_ID, 
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    
    # 6. Trainer
    # We use Reddit Test as the primary eval set during training for early stopping,
    # as it represents the "Ground Truth".
    
    training_args = get_training_args(
        output_dir=OUTPUT_DIR, 
        model_id=STUDENT_MODEL_ID, 
        num_epochs=NUM_EPOCHS,
        train_size=len(tokenized_train)
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_reddit_test, # Optimize for Gold Labels
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("--- Starting Training ---")
    trainer.train()
    
    print(f"--- Saving Student Model to {OUTPUT_DIR} ---")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 7. Final Comprehensive Evaluation
    print("--- Running Final Evaluation on Split Datasets ---")
    
    # Helper to log metrics with prefix
    def log_metrics(dataset, prefix):
        print(f"Evaluating on {prefix}...")
        metrics = trainer.evaluate(dataset)
        # Rename keys to have prefix
        new_metrics = {f"{prefix}_{k.replace('eval_', '')}": v for k, v in metrics.items()}
        wandb.log(new_metrics)
        print(f"{prefix} Metrics: {new_metrics}")
        return new_metrics

    log_metrics(tokenized_wildchat_test, "eval_wildchat")
    log_metrics(tokenized_reddit_test, "eval_reddit")
    
    wandb.finish()

if __name__ == "__main__":
    main()

