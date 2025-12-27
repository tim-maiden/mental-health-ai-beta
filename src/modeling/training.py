import os
import numpy as np
import evaluate
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    TrainingArguments,
    Trainer
)

# Suppress Warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs.get("labels") usually contains the binary/hard labels
        # We need the soft labels. The dataset mapper should have put them in "soft_labels"
        # However, the DataCollator might rename columns or strict checking might apply.
        # Ideally, we mapped 'soft_label' to 'labels' in the tokenization step if we want standard behavior.
        
        # We'll assume inputs["labels"] IS the soft distribution (Float Tensor)
        # The training script must ensure this mapping happens.
        
        labels = inputs.get("labels") # Expected: (batch, num_classes) probabilities
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # KL Divergence Loss
        # Input: Log_Softmax(logits)
        # Target: Probabilities (labels)
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        # nn.KLDivLoss expects input to be log-probabilities and target to be probabilities.
        # batchmean is mathematically correct for KL.
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss = loss_fct(log_probs, labels)
        
        return (loss, outputs) if return_outputs else loss

def get_training_args(output_dir, num_epochs=3, train_batch_size=16, eval_batch_size=16, model_id=None, train_size=0):
    """
    Returns TrainingArguments based on hardware availability.
    """
    is_cloud = os.getenv("DEPLOY_ENV") in ["runpod", "cloud"]
    
    if is_cloud:
        print("--- CONFIGURING FOR CLOUD GPU (H100/A100) ---")
        # [FIX] Increased batch size to saturate H100 memory
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
        evaluation_strategy="epoch",  # Compatible with transformers 4.40.2 (required by optimum 1.19.2)
        save_strategy="epoch",
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.1,  # Increased for better regularization
        lr_scheduler_type="cosine",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        fp16=fp16_mode,
        bf16=bf16_mode,
        dataloader_pin_memory=pin_memory,
        dataloader_num_workers=dataloader_workers,
        logging_steps=50,
        report_to="wandb",
        run_name=f"{model_id.split('/')[-1] if model_id else 'model'}-{train_size}samples",
        
        # [FIX] H100 Optimizations
        # Changed from "reduce-overhead" to "default" to avoid CUDAGraphs memory access errors
        # "default" mode still provides significant speedups via TorchInductor without strict memory constraints
        torch_compile=is_cloud,
        torch_compile_mode="default" if is_cloud else None,
        tf32=is_cloud,  # Enable TensorFloat-32 for significant speedup
    )

def compute_metrics(eval_pred):
    # This metric function is designed for BINARY classification.
    # For Multilabel/Soft-Label, we need to adapt.
    # Assuming the first half of classes are "Safe" (or mapping provided).
    # Actually, simpler: We calculate Top-1 Accuracy against the Soft Label's ArgMax.
    
    predictions, labels = eval_pred
    # Predictions: (batch, num_classes) logits
    # Labels: (batch, num_classes) probabilities
    
    # 1. Hard Accuracy (Did we predict the highest probability class?)
    pred_ids = np.argmax(predictions, axis=1)
    label_ids = np.argmax(labels, axis=1)
    
    accuracy_metric = evaluate.load("accuracy")
    acc = accuracy_metric.compute(predictions=pred_ids, references=label_ids)
    
    return {
        "accuracy": acc["accuracy"]
    }

