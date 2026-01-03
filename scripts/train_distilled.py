import os
import sys
import pandas as pd
import numpy as np
import torch
import wandb
from datasets import Dataset, load_dataset
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
    DISTILLATION_OUTPUT_DIR
)

# Config
# INPUT_FILE = os.path.join(DATA_DIR, "lmsys_silver_labels.pkl")
INPUT_FILE = os.path.join(DATA_DIR, "wildchat_silver_labels.pkl")
OUTPUT_DIR = DISTILLATION_OUTPUT_DIR
NUM_EPOCHS = DISTILLATION_EPOCHS

# Default to the Hub ID, but allow local override
DEFAULT_DATASET_ID = "tim-maiden/mental-health-silver-labels"

def main():
    print(f"--- Starting Student Distillation (Model: {STUDENT_MODEL_ID}) ---")
    
    # 1. WandB Init
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])
    
    try:
        wandb.init(project=WANDB_PROJECT, name="student-distillation", settings=wandb.Settings(init_timeout=300))
    except:
        print("WandB init failed, using offline mode.")
        wandb.init(project=WANDB_PROJECT, mode="offline", name="student-distillation")

    # 2. Load Data (Smart Switching)
    # Check if a command line arg provided a file, otherwise use HF
    dataset_id = DEFAULT_DATASET_ID
    
    print(f"Loading Silver Labels from: {dataset_id}...")
    
    dataset = None
    try:
        # Try loading from Hub
        dataset = load_dataset(dataset_id, split="train")
        print(f"Loaded {len(dataset)} rows from Hugging Face.")
        
        # Rename 'full_dist' to 'labels' to match Trainer expectation
        if "labels" not in dataset.column_names and "full_dist" in dataset.column_names:
            dataset = dataset.rename_column("full_dist", "labels")
            
    except Exception as e:
        print(f"Failed to load from Hub ({e}). Falling back to local file path...")
        # Fallback to legacy local file logic
        if not os.path.exists(INPUT_FILE):
             print(f"Error: Local file {INPUT_FILE} not found.")
             sys.exit(1)
        df = pd.read_pickle(INPUT_FILE)
        df['labels'] = df['full_dist']
        dataset = Dataset.from_pandas(df[['text', 'labels']])

    # Split Train/Test
    # Fix: HuggingFace datasets return a DatasetDict on split
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset_split["train"]
    test_ds = dataset_split["test"]
    
    # 3. Tokenizer
    print(f"Loading Tokenizer ({STUDENT_MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_ds = dataset_split.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 4. Model
    # Retrieve number of labels from the first example of the training set
    example_labels = dataset_split['train'][0]['labels']
    num_labels = len(example_labels)
    print(f"Initializing Model with {num_labels} labels...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL_ID, 
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    
    # 5. Trainer
    training_args = get_training_args(
        output_dir=OUTPUT_DIR, 
        model_id=STUDENT_MODEL_ID, 
        num_epochs=NUM_EPOCHS,
        train_size=len(dataset_split['train'])
    )
    
    # We use the same CustomTrainer which implements KLDivLoss
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("--- Starting Training ---")
    trainer.train()
    
    print(f"--- Saving Student Model to {OUTPUT_DIR} ---")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    wandb.finish()

if __name__ == "__main__":
    main()

