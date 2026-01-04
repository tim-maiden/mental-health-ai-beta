#!/bin/bash
set -e

# --- CONFIGURATION ---
DATASET_ID="tim-maiden/mental-health-silver-labels"
STUDENT_MODEL_REPO="tim-maiden/mental-health-ai-models"
STUDENT_SUBFOLDER="student_deberta_xsmall_v1"

# Teacher Model Configuration (for Inference Step)
TEACHER_MODEL_REPO="tim-maiden/mental-health-ai-models"
# Assuming we use the large model for cloud/production
TEACHER_SUBFOLDER="risk_classifier_deberta_large_v1"

echo "=================================================="
echo "   STARTING DISTILLATION JOB (RUNPOD)"
echo "=================================================="

# 1. Environment Setup
echo "--- Step 1: Environment Check ---"
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not found. Cannot pull dataset or push model."
    exit 1
fi

# 2. Run Inference (Generate Silver Labels)
echo "--- Step 2: Running Inference (Silver Label Generation) ---"

# Check if dataset exists on Hugging Face to avoid redundant computation
echo "Checking availability of dataset: $DATASET_ID"
DATASET_EXISTS=$(python -c "
import os
import sys
from huggingface_hub import HfApi
try:
    token = os.environ.get('HF_TOKEN')
    if not token:
        print('Warning: HF_TOKEN not found in python env', file=sys.stderr)
    api = HfApi(token=token)
    # Check if dataset exists. Throws error if not found or unauthorized.
    # api.dataset_info(repo_id='${DATASET_ID}', repo_type='dataset')
    # Fix: remove repo_type argument which is causing the error
    api.dataset_info(repo_id='${DATASET_ID}')
    print('true')
except Exception as e:
    print(f'Debug: Failed to find dataset {e}', file=sys.stderr)
    print('false')
")

if [ "$DATASET_EXISTS" == "true" ]; then
    echo "✅ Dataset '$DATASET_ID' found on Hugging Face."
    echo "   Skipping inference step (Silver Label Generation)."
else
    echo "⚠️ Dataset '$DATASET_ID' not found."
    echo "   Starting inference to generate silver labels..."
    
    # We run inference on WildChat data using the Teacher Model from HF
    # We limit to 100k samples to keep runtime reasonable (adjust as needed)
    python scripts/inference.py \
        --wildchat \
        --limit 100000 \
        --batch-size 32 \
        --upload-dataset \
        --dataset-id "$DATASET_ID" \
        --model "$TEACHER_MODEL_REPO" \
        --subfolder "$TEACHER_SUBFOLDER"
fi

# 3. Run Training (Distillation)
echo "--- Step 3: Training Student Model ---"
python scripts/train_distilled.py 

# 4. Upload Result
echo "--- Step 4: Uploading Student Model to HF ---"
# We reuse your existing upload script, pointing to the output directory defined in config.py
# We dynamically import the path from config.py to ensure consistency with where the model was saved

python -c "
from huggingface_hub import HfApi
import os
import sys

# Add current directory to path so we can import src
sys.path.append(os.getcwd())

from src.config import DISTILLATION_OUTPUT_DIR

api = HfApi()
print(f'Uploading from: {DISTILLATION_OUTPUT_DIR}')

api.upload_folder(
    folder_path=DISTILLATION_OUTPUT_DIR,
    repo_id='${STUDENT_MODEL_REPO}',
    path_in_repo='${STUDENT_SUBFOLDER}',
    ignore_patterns=['checkpoint-*', 'runs/*']
)
print('Upload complete.')
"

# 4. Cleanup & Termination
echo "--- Step 4: Job Complete. Terminating Pod. ---"
if [ -f "scripts/terminate_pod_remote.sh" ]; then
    bash scripts/terminate_pod_remote.sh
else
    echo "Warning: Termination script not found."
fi

