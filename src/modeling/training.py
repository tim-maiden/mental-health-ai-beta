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

def get_training_args(output_dir, num_epochs=3, train_batch_size=16, eval_batch_size=16, model_id=None, train_size=0):
    """
    Returns TrainingArguments based on hardware availability.
    """
    is_cloud = os.getenv("DEPLOY_ENV") in ["runpod", "cloud"]
    
    if is_cloud:
        print("--- CONFIGURING FOR CLOUD GPU (H100/A100) ---")
        # DeBERTa Large requires smaller batch sizes due to memory constraints
        # Reduced from 32 to 16 to accommodate larger model
        per_device_train_batch_size = 16
        per_device_eval_batch_size = 16
        grad_accum_steps = 2  # Increased to maintain effective batch size
        bf16_mode = True
        fp16_mode = False
        dataloader_workers = 8
        pin_memory = True
    else:
        print("--- CONFIGURING FOR LOCAL (MAC/MPS) ---")
        per_device_train_batch_size = 4
        per_device_eval_batch_size = 4
        grad_accum_steps = 4
        bf16_mode = False
        fp16_mode = False # MPS crash workaround
        dataloader_workers = 0
        pin_memory = False

    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",  # Compatible with transformers 4.40.2 (required by optimum 1.19.2)
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
        bf16=bf16_mode,
        dataloader_pin_memory=pin_memory,
        dataloader_num_workers=dataloader_workers,
        logging_steps=50,
        report_to="wandb",
        run_name=f"{model_id.split('/')[-1] if model_id else 'model'}-{train_size}samples",
        torch_compile=is_cloud,  # Enable native torch.compile via Trainer (H100 optimization)
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
    
    # Add mean probability if available (for calibration monitoring)
    metrics = {
        "accuracy": acc["accuracy"],
        "precision": prec["precision"],
        "recall": rec["recall"],
        "f1": f1["f1"]
    }
    
    if hasattr(eval_pred, "predictions") and isinstance(eval_pred.predictions, np.ndarray):
        # eval_pred.predictions is logits. Apply softmax to get probabilities.
        # Check dimensions. If (n_samples, n_classes), we want prob of class 1.
        
        logits = eval_pred.predictions
        # Simple softmax
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        # Avg prob of Risk (class 1)
        avg_risk_prob = np.mean(probs[:, 1])
        metrics["avg_risk_prob"] = avg_risk_prob
        
    return metrics

