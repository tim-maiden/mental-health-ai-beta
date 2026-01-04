#!/bin/bash
set -e

# --- CONFIGURATION ---
WILDCHAT_DATASET_ID="tim-maiden/mental-health-silver-wildchat"
REDDIT_SILVER_DATASET_ID="tim-maiden/mental-health-silver-reddit"
STUDENT_MODEL_REPO="tim-maiden/mental-health-ai-models"
STUDENT_SUBFOLDER="student_deberta_small_v1"

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
echo "--- Step 2a: Inference on WildChat ---"

# Check if WildChat Silver exists
echo "Checking availability of dataset: $WILDCHAT_DATASET_ID"
WILDCHAT_EXISTS=$(python -c "
import os
import sys
from huggingface_hub import HfApi
try:
    token = os.environ.get('HF_TOKEN')
    api = HfApi(token=token)
    api.dataset_info(repo_id='${WILDCHAT_DATASET_ID}')
    print('true')
except Exception as e:
    print('false')
")

if [ "$WILDCHAT_EXISTS" == "true" ]; then
    echo "✅ Dataset '$WILDCHAT_DATASET_ID' found. Skipping WildChat inference."
else
    echo "⚠️ Dataset '$WILDCHAT_DATASET_ID' not found. Starting inference..."
    python scripts/inference.py \
        --wildchat \
        --batch-size 32 \
        --upload-dataset \
        --dataset-id "$WILDCHAT_DATASET_ID" \
        --model "$TEACHER_MODEL_REPO" \
        --subfolder "$TEACHER_SUBFOLDER"
fi

echo "--- Step 2b: Inference on Reddit ---"

# Check if Reddit Silver exists
echo "Checking availability of dataset: $REDDIT_SILVER_DATASET_ID"
REDDIT_EXISTS=$(python -c "
import os
import sys
from huggingface_hub import HfApi
try:
    token = os.environ.get('HF_TOKEN')
    api = HfApi(token=token)
    api.dataset_info(repo_id='${REDDIT_SILVER_DATASET_ID}')
    print('true')
except Exception as e:
    print('false')
")

if [ "$REDDIT_EXISTS" == "true" ]; then
    echo "✅ Dataset '$REDDIT_SILVER_DATASET_ID' found. Skipping Reddit inference."
else
    echo "⚠️ Dataset '$REDDIT_SILVER_DATASET_ID' not found. Starting inference..."
    # We downsample Reddit inference to 1M to match WildChat size roughly, saving compute
    # This gives us ~2M total training rows (1M WildChat + 1M Reddit)
    python scripts/inference.py \
        --reddit \
        --limit 1000000 \
        --batch-size 32 \
        --upload-dataset \
        --dataset-id "$REDDIT_SILVER_DATASET_ID" \
        --model "$TEACHER_MODEL_REPO" \
        --subfolder "$TEACHER_SUBFOLDER"
fi

# 3. Run Training (Distillation)
echo "--- Step 3: Training Student Model (All-Silver Strategy) ---"
python scripts/train_mixed_student.py 

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

