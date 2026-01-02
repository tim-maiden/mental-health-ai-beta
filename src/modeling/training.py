import os
import numpy as np
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    TrainingArguments,
    Trainer
)

# Suppress Warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

class CustomTrainer(Trainer):
    def __init__(self, risk_indices=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store the indices of the "Risk" classes (e.g., [0, 4, 12...])
        self.risk_indices = torch.tensor(risk_indices) if risk_indices else None

    def compute_loss(self, model, inputs, return_outputs=False):
        # Assumes 'inputs["labels"]' contains the soft target distribution (Float Tensor), mapped during tokenization.
        
        labels = inputs.get("labels") # Expected: (batch, num_classes) probabilities
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 1. Standard Distillation Loss (Fine-Grained)
        # Input: Log_Softmax(logits)
        # Target: Probabilities (labels)
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        # nn.KLDivLoss expects input to be log-probabilities and target to be probabilities.
        # batchmean is mathematically correct for KL.
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        main_loss = loss_fct(log_probs, labels)
        
        # 2. Binary Consistency Loss (Hierarchical Loss)
        if self.risk_indices is not None:
            # Ensure indices are on correct device
            if self.risk_indices.device != logits.device:
                self.risk_indices = self.risk_indices.to(logits.device)

            # Calculate total probability mass predicted for RISK classes
            probs = F.softmax(logits, dim=-1)
            pred_risk_mass = torch.sum(probs[:, self.risk_indices], dim=1)
            
            # Calculate total probability mass the TEACHER assigned to RISK classes
            target_risk_mass = torch.sum(labels[:, self.risk_indices], dim=1)
            
            # Force them to match (MSE)
            aux_loss = F.mse_loss(pred_risk_mass, target_risk_mass)
            
            # Combine: 80% Fine-Grained, 20% Binary enforcement (0.5 weight relative to KL roughly)
            total_loss = main_loss + (0.5 * aux_loss)
        else:
            total_loss = main_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

from src.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    SAVE_TOTAL_LIMIT
)

def get_training_args(output_dir, num_epochs=3, train_batch_size=16, eval_batch_size=16, model_id=None, train_size=0):
    """
    Returns TrainingArguments based on hardware availability.
    """
    is_cloud = os.getenv("DEPLOY_ENV") in ["runpod", "cloud"]
    
    if is_cloud:
        print("--- CONFIGURING FOR CLOUD GPU (H100/A100) ---")
        # Increase batch size to saturate H100 memory
        per_device_train_batch_size = 32
        per_device_eval_batch_size = 32
        grad_accum_steps = 2  
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,  # Increased for better regularization
        lr_scheduler_type="cosine",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        save_total_limit=SAVE_TOTAL_LIMIT,  # Only keep the best checkpoint to prevent disk space exhaustion
        fp16=fp16_mode,
        bf16=bf16_mode,
        dataloader_pin_memory=pin_memory,
        dataloader_num_workers=dataloader_workers,
        logging_steps=50,
        report_to="wandb",
        run_name=f"{model_id.split('/')[-1] if model_id else 'model'}-{train_size}samples",
        
        # H100 Optimization: Use "default" mode for torch_compile to prevent CUDAGraphs memory errors.
        torch_compile=is_cloud,
        torch_compile_mode="default" if is_cloud else None,
        tf32=is_cloud,  # Enable TensorFloat-32 for significant speedup
        dataloader_persistent_workers=is_cloud,
        ddp_find_unused_parameters=False if is_cloud else None,
    )

def compute_metrics(eval_pred):
    # This metric function is designed for BINARY classification.
    # For Multilabel/Soft-Label, we need to adapt.
    # Calculate Top-1 Accuracy against the ArgMax of the soft label distribution.
    
    predictions, labels = eval_pred
    # Predictions: (batch, num_classes) logits
    # Labels: (batch, num_classes) probabilities
    
    # 1. Hard Accuracy (Did we predict the highest probability class?)
    pred_ids = np.argmax(predictions, axis=1)
    label_ids = np.argmax(labels, axis=1)
    
    # Standard Metrics (Weighted for class imbalance)
    accuracy = accuracy_score(label_ids, pred_ids)
    precision, recall, f1, _ = precision_recall_fscore_support(
        label_ids, 
        pred_ids, 
        average='weighted', 
        zero_division=0
    )
    
    # 2. Top-3 Accuracy
    # Did the correct label appear in the top 3 predictions?
    top3_accuracy = accuracy # Fallback
    if predictions.shape[1] >= 3:
        # Get indices of top 3 logits
        top3_preds = np.argsort(predictions, axis=1)[:, -3:]
        # Check if label_ids is in top3_preds
        # Reshape label_ids to (N, 1) for broadcasting
        correct_top3 = np.any(top3_preds == label_ids[:, np.newaxis], axis=1)
        top3_accuracy = np.mean(correct_top3)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "top3_accuracy": top3_accuracy
    }

