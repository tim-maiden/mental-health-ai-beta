import os
import sys
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer

# Disable tokenizer parallelism to prevent deadlocks with DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.modeling.training import get_training_args, compute_metrics
from src.config import (
    MODEL_ID, 
    TRAIN_FILE, 
    TEST_CLEAN_FILE, 
    TEST_AMBIGUOUS_FILE, 
    MODEL_OUTPUT_DIR, 
    WANDB_PROJECT
)

EVAL_FILES = {
    "clean": TEST_CLEAN_FILE,
    "ambiguous": TEST_AMBIGUOUS_FILE
}     
OUTPUT_DIR = MODEL_OUTPUT_DIR
NUM_EPOCHS = 5

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

    dataset = load_dataset("json", data_files={
        "train": TRAIN_FILE,
        "test_clean": EVAL_FILES["clean"],
        "test_ambiguous": EVAL_FILES["ambiguous"]
    })
    
    print(f"--- Loading Tokenizer ({MODEL_ID}) ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    def preprocess_function(examples):
        # Tokenize
        tokenized = tokenizer(examples["text"], truncation=True, max_length=512)
        # Rename 'label' to 'labels' to match Model signature exactly
        tokenized["labels"] = examples["label"]
        return tokenized

    # CRITICAL FIX: Remove the raw 'text' and old 'label' columns so Trainer doesn't get confused
    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=["text", "label", "subreddit"] if "subreddit" in dataset["train"].column_names else ["text", "label"]
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"--- Initializing Model ---")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, 
        num_labels=2,
        label2id={"Safe": 0, "Risk": 1},
        id2label={0: "Safe", 1: "Risk"}
    )

    # Note: Model compilation is now handled by Trainer via torch_compile parameter
    # This avoids signature inspection issues with torch.compile()

    training_args = get_training_args(
        output_dir=OUTPUT_DIR,
        model_id=MODEL_ID,
        train_size=len(dataset['train'])
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset={
            "clean": tokenized_datasets["test_clean"],
            "ambiguous": tokenized_datasets["test_ambiguous"]
        },
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("--- Starting Training ---")
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    trainer.train()

    print(f"--- Saving Model to {OUTPUT_DIR} ---")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n--- Final Evaluation ---")
    print("Evaluating on Clean Test Set...")
    print(trainer.evaluate(eval_dataset=tokenized_datasets["test_clean"], metric_key_prefix="eval_clean"))

    print("Evaluating on Ambiguous Test Set...")
    print(trainer.evaluate(eval_dataset=tokenized_datasets["test_ambiguous"], metric_key_prefix="eval_ambiguous"))
    
    wandb.finish()

if __name__ == "__main__":
    main()
