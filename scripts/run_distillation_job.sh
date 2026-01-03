#!/bin/bash
set -e

# --- CONFIGURATION ---
DATASET_ID="tim-maiden/mental-health-silver-labels"
STUDENT_MODEL_REPO="tim-maiden/mental-health-ai-models"
STUDENT_SUBFOLDER="student_distilbert_v1"

echo "=================================================="
echo "   STARTING DISTILLATION JOB (RUNPOD)"
echo "=================================================="

# 1. Environment Setup
echo "--- Step 1: Environment Check ---"
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not found. Cannot pull dataset or push model."
    exit 1
fi

# 2. Run Training
# We don't need to run inference first because we assume Phase 1 is done.
echo "--- Step 2: Training Student Model ---"
python scripts/train_distilled.py 

# 3. Upload Result
echo "--- Step 3: Uploading Student Model to HF ---"
# We reuse your existing upload script, pointing to the output directory defined in config.py
# (DISTILLATION_OUTPUT_DIR = "models/final_student_distilbert")

# We use the explicit upload command to ensure it goes to the right subfolder
python -c "
from huggingface_hub import HfApi
import os
api = HfApi()
api.upload_folder(
    folder_path='models/final_student_distilbert',
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

