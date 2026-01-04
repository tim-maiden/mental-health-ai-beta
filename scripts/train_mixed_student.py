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
WILDCHAT_SILVER_ID = "tim-maiden/mental-health-silver-wildchat"
REDDIT_SILVER_ID = "tim-maiden/mental-health-silver-reddit"
OUTPUT_DIR = DISTILLATION_OUTPUT_DIR
NUM_EPOCHS = DISTILLATION_EPOCHS

def prepare_silver_dataset(dataset):
    """
    Standardizes a silver dataset (renaming columns).
    """
    if "labels" not in dataset.column_names and "full_dist" in dataset.column_names:
        dataset = dataset.rename_column("full_dist", "labels")
    
    # Ensure text column is named 'text'
    if "input" in dataset.column_names and "text" not in dataset.column_names:
        dataset = dataset.rename_column("input", "text")
        
    return dataset

def main():
    print(f"--- Starting Student Distillation (Model: {STUDENT_MODEL_ID}) ---")
    print(f"--- Strategy: All-Silver (WildChat Silver + Reddit Silver) ---")
    
    # 1. WandB Init
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])
    
    try:
        wandb.init(project=WANDB_PROJECT, name="student-distillation-all-silver", settings=wandb.Settings(init_timeout=300))
    except:
        print("WandB init failed, using offline mode.")
        wandb.init(project=WANDB_PROJECT, mode="offline", name="student-distillation-all-silver")

    # 2. Load WildChat Silver
    print(f"Loading WildChat Silver from: {WILDCHAT_SILVER_ID}...")
    wildchat_ds = load_dataset(WILDCHAT_SILVER_ID, split="train")
    wildchat_ds = prepare_silver_dataset(wildchat_ds)
    
    # Split WildChat
    wildchat_split = wildchat_ds.train_test_split(test_size=0.1, seed=42)
    wildchat_train = wildchat_split["train"]
    wildchat_test = wildchat_split["test"]
    
    # 3. Load Reddit Silver
    print(f"Loading Reddit Silver from: {REDDIT_SILVER_ID}...")
    reddit_ds = load_dataset(REDDIT_SILVER_ID, split="train") 
    reddit_ds = prepare_silver_dataset(reddit_ds)

    # Determine num_labels from WildChat
    example_labels = wildchat_train[0]['labels']
    num_labels = len(example_labels)
    print(f"Detected {num_labels} classes from Silver labels.")

    # Split Reddit
    reddit_split = reddit_ds.train_test_split(test_size=0.1, seed=42)
    reddit_train = reddit_split["train"]
    reddit_test = reddit_split["test"]
    
    # Balancing Logic - SKIPPED to avoid data loss
    # We simply concatenate all available data. Transformers are robust to mild imbalance.
    print(f"Skipping Downsampling. Using full datasets.")
    
    print(f"Data Stats:")
    print(f"  WildChat Train: {len(wildchat_train)}")
    print(f"  Reddit Train:   {len(reddit_train)}")
    print(f"  WildChat Test:  {len(wildchat_test)}")
    print(f"  Reddit Test:    {len(reddit_test)}")
    
    # 4. Concatenate Training Data
    combined_train = concatenate_datasets([wildchat_train, reddit_train])
    combined_train = combined_train.shuffle(seed=42) 
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
        eval_dataset=tokenized_reddit_test, # Default eval (will be overridden by custom loop)
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
    
    def log_metrics(dataset, prefix):
        print(f"Evaluating on {prefix}...")
        metrics = trainer.evaluate(dataset)
        new_metrics = {f"{prefix}_{k.replace('eval_', '')}": v for k, v in metrics.items()}
        wandb.log(new_metrics)
        print(f"{prefix} Metrics: {new_metrics}")
        return new_metrics

    log_metrics(tokenized_wildchat_test, "eval_wildchat")
    log_metrics(tokenized_reddit_test, "eval_reddit")
    
    wandb.finish()

if __name__ == "__main__":
    main()

