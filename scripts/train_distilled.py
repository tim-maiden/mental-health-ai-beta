import os
import sys
import pandas as pd
import numpy as np
import torch
import wandb
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.modeling.training import get_training_args, CustomTrainer, compute_metrics
from src.config import (
    MODEL_OUTPUT_DIR, 
    WANDB_PROJECT,
    DATA_DIR
)

# Config
STUDENT_MODEL_ID = "distilbert-base-uncased"
INPUT_FILE = os.path.join(DATA_DIR, "lmsys_silver_labels.pkl")
OUTPUT_DIR = "models/final_student_distilbert"
NUM_EPOCHS = 5

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

    # 2. Load Silver Labels
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Run inference.py first.")
        sys.exit(1)
        
    print(f"Loading silver labels from {INPUT_FILE}...")
    df = pd.read_pickle(INPUT_FILE)
    
    # Filter for confidence? (Optional)
    # The expert mentioned this but didn't mandate it. 
    # High confidence samples are better teachers.
    # Let's keep all for now as the Soft Labels carry the uncertainty info.
    
    # Ensure 'labels' column contains the 'full_dist' list
    # Convert list to float32 numpy array/tensor logic is handled by collator/trainer usually,
    # but safe to keep as list of floats in the dataset.
    df['labels'] = df['full_dist']
    
    # Convert to HF Dataset
    dataset = Dataset.from_pandas(df[['text', 'labels']])
    
    # Split Train/Test
    dataset = dataset.train_test_split(test_size=0.1)
    
    # 3. Tokenizer
    print(f"Loading Tokenizer ({STUDENT_MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_ds = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 4. Model
    num_labels = len(df['labels'].iloc[0])
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
        train_size=len(dataset['train'])
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

