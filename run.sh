#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# --- Defaults ---
CLEAN=true
DEPLOY_ENV="local"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="run_${TIMESTAMP}.log"

# Redirect all output to log file while keeping stdout
exec > >(tee -a "$LOG_FILE") 2>&1

# --- Helpers ---
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

upload_logs() {
    if [ "$DEPLOY_ENV" == "runpod" ] || [ "$DEPLOY_ENV" == "cloud" ]; then
        log "Uploading logs to S3..."
        # Upload using the python script, putting logs in a timestamped folder
        python scripts/upload_logs.py --log-file "$LOG_FILE" --s3-prefix "logs/$TIMESTAMP" || log "Warning: Failed to upload logs."
    fi
}

error_handler() {
    log "Error occurred in script at line: $1"
    log "Last command exit code: $?"
    log "Pipeline failed."
    upload_logs
    exit 1
}

trap 'error_handler $LINENO' ERR

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-clean) CLEAN=false ;;
        --deploy) DEPLOY_ENV="$2"; shift ;;
        *) log "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

log "--- Starting Pipeline (Mode: $DEPLOY_ENV) ---"
export DEPLOY_ENV=$DEPLOY_ENV

# 1. Clean previous artifacts
if [ "$CLEAN" = true ]; then
    log "Cleaning previous artifacts..."
    if [ -f "./scripts/cleanup.sh" ]; then
        ./scripts/cleanup.sh
    else
        log "Warning: cleanup.sh not found, skipping."
    fi
fi

# 2. Setup Environment
if [ "$DEPLOY_ENV" == "local" ]; then
    # --- LOCAL (Mac/MPS) ---
    if [ ! -d "venv" ]; then
        log "Creating virtual environment..."
        python3.12 -m venv venv
        source venv/bin/activate
        
        log "Installing dependencies (Local/Mac)..."
        if [ -f "requirements_local.txt" ]; then
             pip install -r requirements_local.txt
        else
             log "Warning: requirements_local.txt not found, falling back to requirements.txt"
             pip install -r requirements.txt
        fi
    else
        source venv/bin/activate
    fi
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

elif [ "$DEPLOY_ENV" == "runpod" ] || [ "$DEPLOY_ENV" == "cloud" ]; then
    # --- CLOUD (CUDA) ---
    log "Configuring Cloud Environment..."
    
    pip install --upgrade pip --break-system-packages
    pip install -r requirements.txt --break-system-packages
    unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
else
    log "Error: Invalid deploy mode. Use 'local' or 'runpod'."
    exit 1
fi

# 3. Pipeline Steps
export HF_HUB_ENABLE_HF_TRANSFER=0

# Check for compiled data first to avoid expensive re-ingestion
log "--- Checking for pre-compiled datasets ---"
if python scripts/download_datasets.py --s3-prefix "data/latest"; then
    log "Successfully downloaded compiled datasets from S3. Skipping Data Ingestion and Compilation."
else
    log "Compiled data not found or download failed. Proceeding with full data pipeline..."

    # Step 1: Data Ingestion (Snapshot)
    # Skipped on RunPod to enforce "Freeze" workflow.
    # Ensure you have run 'python scripts/ingest_data.py' locally first!
    log "--- Step 1: Data Ingestion (Skipped - Reading from S3 Snapshot) ---"
    # python scripts/ingest_data.py

    # Step 2: Compile Dataset (Teacher Training Data)
    log "--- Step 2: Dataset Compilation (Soft Labels via k-NN) ---"
    python scripts/compile_dataset.py

    # [NEW] Step 2.5: Backup Compiled Data
    log "--- Uploading Compiled Data to S3 ---"
    
    # Create temp dir for specific artifacts we want to version (to avoid uploading raw pickles)
    mkdir -p data/compiled_artifacts
    cp data/final_train.parquet data/compiled_artifacts/ 2>/dev/null || true
    cp data/test.parquet data/compiled_artifacts/ 2>/dev/null || true
    cp data/subreddit_mapping.json data/compiled_artifacts/ 2>/dev/null || true
    
    # Upload to timestamped folder (Versioning for reproducibility)
    python scripts/upload_model.py --local-dir "data/compiled_artifacts" --s3-prefix "data/${TIMESTAMP}"
    
    # Upload to 'latest' folder (Caching for speed)
    python scripts/upload_model.py --local-dir "data/compiled_artifacts" --s3-prefix "data/latest"
    
    # Cleanup
    rm -rf data/compiled_artifacts
fi

# Step 3: Train Teacher Model (DeBERTa)
log "--- Step 3: Train Teacher (DeBERTa) ---"
# This trains the model that learns to imitate the Reddit k-NN distribution
python scripts/train_classifier.py

# Step 4: Inference on Target Domain (WildChat - Silver Labeling)
# log "--- Step 4: Generate Silver Labels (WildChat via Supabase) ---"
# # Run inference on WildChat data from Supabase to generate soft labels
# # Save output to data/wildchat_silver_labels.pkl
# # Use --limit 50000 or however many rows you want to label for distillation
# python scripts/inference.py --wildchat --output data/wildchat_silver_labels.pkl --limit 50000

# Step 5: Train Student Model (Distillation)
# log "--- Step 5: Train Student (DistilBERT/MobileBERT) ---"
# python scripts/train_distilled.py

log "--- Step 6: Final Inference Test ---"
python scripts/inference.py

log "--- Step 7: Upload to S3 ---"
if [ "$DEPLOY_ENV" == "runpod" ] || [ "$DEPLOY_ENV" == "cloud" ]; then
    MODELS_LOCAL_DIR="/workspace/models"
else
    MODELS_LOCAL_DIR="models"
fi
python scripts/upload_model.py --local-dir "$MODELS_LOCAL_DIR" --s3-prefix models

# Upload logs at the end of a successful run
upload_logs

log "--- Pipeline Complete! ---"

# Step 8: Terminate pod if running on RunPod
if [ "$DEPLOY_ENV" == "runpod" ] || [ "$DEPLOY_ENV" == "cloud" ]; then
    log "--- Step 8: Terminating Pod ---"
    if [ -f "./scripts/terminate_pod_remote.sh" ]; then
        bash ./scripts/terminate_pod_remote.sh || log "Warning: Pod termination script failed, but pipeline completed successfully."
    else
        log "Warning: terminate_pod.sh not found. Pod will continue running."
        log "You may need to terminate it manually via the RunPod console or CLI."
    fi
fi
