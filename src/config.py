import json
import os
import sys
from dotenv import load_dotenv

# --- CONFIGURATION ---
# Load environment variables from .env file if present
load_dotenv()

# 1. Load credentials from Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET")

# 2. Validate Critical Credentials
if not OPENAI_API_KEY:
    print("Warning: OPENAI_KEY not found in environment variables")
if not SUPABASE_URL or not SUPABASE_KEY:
    print("Warning: Supabase credentials not found in environment variables")


# --- AWS CONFIGURATION ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    print("Warning: AWS credentials not found. S3 upload will be disabled.")


# --- PROCESSING CONFIGURATION ---
BATCH_SIZE = 500  # Reduced from 1000 to avoid Supabase statement timeouts
PROGRESS_FILE = "progress.json"

# --- API LIMITS CONFIGURATION ---
TPM_LIMIT = 5_000_000
RPM_LIMIT = 5_000

# --- MODEL & TRAINING CONFIGURATION ---
# Environment-aware model selection:
# - Local: DeBERTa Small (faster, lower memory)
# - Cloud/RunPod: DeBERTa Large (better accuracy, more GPU resources)
DEPLOY_ENV = os.getenv("DEPLOY_ENV", "local")
if DEPLOY_ENV in ["runpod", "cloud"]:
    MODEL_ID = "microsoft/deberta-v3-large"
    MODEL_SIZE = "large"
else:
    MODEL_ID = "microsoft/deberta-v3-small"
    MODEL_SIZE = "small"

WANDB_PROJECT = "mental-health-classifier"

# --- INFERENCE CONFIGURATION ---
# Temperature scaling for confidence calibration
# T < 1.0 sharpens predictions (makes them more confident)
# T = 0.3 is optimized for margin-based classifiers that produce "squashed" probabilities
# Rationale: Model learns correct direction but lacks hard negatives that force high-magnitude logits
TEMPERATURE = 1.0

# Paths
if DEPLOY_ENV in ["runpod", "cloud"]:
    # Use persistent volume in cloud environments to avoid container disk exhaustion
    # RunPod mounts the network volume at /workspace
    BASE_DIR = "/workspace"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    print(f"--- Cloud Environment Detected: Using Persistent Volume at {BASE_DIR} ---")
else:
    # Use local relative paths
    DATA_DIR = "data"
    OUTPUT_DIR = "outputs"
    MODELS_DIR = "models"

SNAPSHOTS_DIR = os.path.join(DATA_DIR, "snapshots")

# Data Files
# Changed to PKL for binary efficiency and type preservation
# RAW_DATA_FILE = os.path.join(DATA_DIR, "raw_latest.pkl")
# UPDATED: Use S3 Parquet to avoid local disk exhaustion
RAW_DATA_FILE = f"s3://{S3_BUCKET_NAME}/data/latest/raw_data.parquet"

TRAIN_FILE = os.path.join(DATA_DIR, "final_train.parquet")
TEST_FILE = os.path.join(DATA_DIR, "test.parquet")

# Model Paths (include model size to avoid conflicts between local/cloud models)
MODEL_OUTPUT_DIR = os.path.join(MODELS_DIR, f"risk_classifier_deberta_{MODEL_SIZE}_v1")
