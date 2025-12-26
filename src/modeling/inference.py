import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import TEMPERATURE

try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
except Exception as e:
    print(f"Warning: Failed to import ORTModelForSequenceClassification: {e}")
    ORTModelForSequenceClassification = None

def get_device():
    device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device_str)

def load_model(model_path, is_quantized=False):
    device = get_device()
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if is_quantized:
        if ORTModelForSequenceClassification is None:
             raise ImportError("ORTModelForSequenceClassification could not be imported. Check your installation.")

        is_cloud = os.getenv("DEPLOY_ENV") in ["runpod", "cloud"]
        providers = ["CUDAExecutionProvider"] if is_cloud else ["CPUExecutionProvider"]
        
        print(f"Loading ONNX Model with providers: {providers}")
        model = ORTModelForSequenceClassification.from_pretrained(
            model_path, 
            file_name="model_quantized.onnx",
            provider=providers[0]
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.to(device)
    return model, tokenizer, device

def predict_batch(model, tokenizer, texts, device, is_quantized=False, temperature=None):
    """
    Runs batch inference with optional temperature scaling to sharpen confidence.
    
    Args:
        model: The trained Transformer model (PyTorch or ONNX)
        tokenizer: Tokenizer for the model
        texts: List of input texts
        device: Torch device
        is_quantized: Whether using ONNX quantized model
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
        if is_quantized:
            logits = model(**inputs).logits
        else:
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

