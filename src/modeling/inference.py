import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import TEMPERATURE

def get_device():
    device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device_str)

def load_model(model_path, **kwargs):
    device = get_device()
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
    
    model.to(device)
    return model, tokenizer, device

def predict_batch(model, tokenizer, texts, device, temperature=None):
    """
    Runs batch inference with optional temperature scaling to sharpen confidence.
    
    Args:
        model: The trained Transformer model
        tokenizer: Tokenizer for the model
        texts: List of input texts
        device: Torch device
        temperature (float): Temperature scaling factor. 
                             T < 1.0 sharpens predictions (makes them more confident).
                             T = 0.3 is optimized for margin-based classifiers.
                             If None, uses TEMPERATURE from config (default 0.3).
    
    Returns:
        Tensor of shape [batch_size, num_classes] with probabilities
    """
    if temperature is None:
        temperature = TEMPERATURE
    
    # 1. Tokenize
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 2. Predict (get raw logits)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # 3. Apply Temperature Scaling
    # Dividing by T < 1.0 increases the magnitude of logits,
    # pushing probabilities closer to 0 or 1 (sharper predictions)
    scaled_logits = logits / temperature
    
    # 4. Convert to Probabilities
    probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
    return probs

def is_clean_english(text):
    if not text or len(text) == 0: 
        return False
    non_ascii = len(re.findall(r'[^\x00-\x7F]', text))
    return (non_ascii / len(text)) < 0.2

