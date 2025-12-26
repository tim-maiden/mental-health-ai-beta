import os
import numpy as np
import evaluate
import warnings
import torch
from torch import nn
from transformers import (
    TrainingArguments,
    Trainer
)

# Suppress Warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

def get_training_args(output_dir, num_epochs=3, train_batch_size=16, eval_batch_size=16, model_id="microsoft/deberta-v3-base", train_size=0):
    """
    Returns TrainingArguments based on hardware availability.
    """
    is_cloud = os.getenv("DEPLOY_ENV") in ["runpod", "cloud"]
    
    if is_cloud:
        print("--- CONFIGURING FOR CLOUD GPU (H100/A100) ---")
        per_device_train_batch_size = 32
        per_device_eval_batch_size = 32
        grad_accum_steps = 1
        fp16_mode = True
        dataloader_workers = 8
        pin_memory = True
    else:
        print("--- CONFIGURING FOR LOCAL (MAC/MPS) ---")
        per_device_train_batch_size = 4
        per_device_eval_batch_size = 4
        grad_accum_steps = 4
        fp16_mode = False # MPS crash workaround
        dataloader_workers = 0
        pin_memory = False

    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",  # Updated from evaluation_strategy for transformers >= 4.41
        save_strategy="epoch",
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.1,  # Increased for better regularization
        lr_scheduler_type="cosine",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_ambiguous_loss",
        fp16=fp16_mode,
        dataloader_pin_memory=pin_memory,
        dataloader_num_workers=dataloader_workers,
        logging_steps=50,
        report_to="wandb",
        run_name=f"{model_id.split('/')[-1]}-{train_size}samples"
    )

def compute_metrics(eval_pred):
    # Load metrics locally to avoid issues
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    prec = precision_metric.compute(predictions=predictions, references=labels)
    rec = recall_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    
    return {
        "accuracy": acc["accuracy"],
        "precision": prec["precision"],
        "recall": rec["recall"],
        "f1": f1["f1"]
    }

