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
        
        log "Installing dependencies..."
        pip install -r requirements.txt
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

# Step 1: Data Ingestion (Snapshot)
log "--- Step 1: Data Ingestion ---"
python scripts/ingest_data.py

# Step 2: Compile Dataset (Teacher Training Data)
log "--- Step 2: Dataset Compilation (Soft Labels via k-NN) ---"
python scripts/compile_dataset.py

# Step 3: Train Teacher Model (DeBERTa)
log "--- Step 3: Train Teacher (DeBERTa) ---"
# This trains the model that learns to imitate the Reddit k-NN distribution
python scripts/train_classifier.py

# Step 4: Inference on Target Domain (LMSYS - Silver Labeling)
log "--- Step 4: Generate Silver Labels (LMSYS) ---"
# Run inference on LMSYS data to generate soft labels (probabilities)
# Save output to data/lmsys_silver_labels.pkl
python scripts/inference.py --lmsys --output data/lmsys_silver_labels.pkl --limit 50000

# Step 5: Train Student Model (Distillation)
log "--- Step 5: Train Student (DistilBERT/MobileBERT) ---"
python scripts/train_distilled.py

# log "--- Step 4: Quantize Model (ONNX FP16) ---"
#
# # Determine model paths based on environment (matching config.py logic)
# if [ "$DEPLOY_ENV" == "runpod" ] || [ "$DEPLOY_ENV" == "cloud" ]; then
#     MODEL_SIZE="large"
# else
#     MODEL_SIZE="small"
# fi
#
# # Dynamic path definitions (must match config.py)
# SOURCE_MODEL_DIR="models/risk_classifier_deberta_${MODEL_SIZE}_v1"
# TARGET_QUANTIZED_DIR="models/risk_classifier_quantized_${MODEL_SIZE}"
#
# log "Source Model: $SOURCE_MODEL_DIR"
# log "Target Quantized Dir: $TARGET_QUANTIZED_DIR"
#
# # Verify source model exists
# if [ ! -d "$SOURCE_MODEL_DIR" ]; then
#     log "Error: Source model directory not found: $SOURCE_MODEL_DIR"
#     log "Please ensure training completed successfully in Step 3."
#     exit 1
# fi
#
# mkdir -p "$TARGET_QUANTIZED_DIR"
#
# # Verify optimum-cli exists
# if ! command -v optimum-cli &> /dev/null; then
#     log "Error: optimum-cli not found. Check installation."
#     exit 1
# fi
#
# log "Exporting to ONNX (FP16)..."
#
# # [FIX] Added --device cuda to support FP16 export (required on GPU)
# DEVICE_OPTS=""
# if [ "$DEPLOY_ENV" == "runpod" ] || [ "$DEPLOY_ENV" == "cloud" ]; then
#     DEVICE_OPTS="--device cuda"
# fi
#
# log "ONNX Export Config:"
# log "  - Model: $SOURCE_MODEL_DIR"
# log "  - Device Opts: '$DEVICE_OPTS'"
# log "  - Output: $TARGET_QUANTIZED_DIR/"
#
# if ! optimum-cli export onnx --model "$SOURCE_MODEL_DIR" --task text-classification --dtype fp16 --opset 17 $DEVICE_OPTS "$TARGET_QUANTIZED_DIR/"; then
#     log "Error: ONNX export failed. Check the model directory and optimum-cli installation."
#     exit 1
# fi
#
# # Rename for Inference Compatibility
# if [ -f "$TARGET_QUANTIZED_DIR/model.onnx" ]; then
#     mv "$TARGET_QUANTIZED_DIR/model.onnx" "$TARGET_QUANTIZED_DIR/model_quantized.onnx"
# fi
#
# # Copy config files
# log "Copying model artifacts..."
# cp "$SOURCE_MODEL_DIR/config.json" "$TARGET_QUANTIZED_DIR/" 2>/dev/null || true
# cp "$SOURCE_MODEL_DIR/tokenizer_config.json" "$TARGET_QUANTIZED_DIR/" 2>/dev/null || true
# cp "$SOURCE_MODEL_DIR/vocab.txt" "$TARGET_QUANTIZED_DIR/" 2>/dev/null || true
# cp "$SOURCE_MODEL_DIR/tokenizer.json" "$TARGET_QUANTIZED_DIR/" 2>/dev/null || true
# cp "$SOURCE_MODEL_DIR/special_tokens_map.json" "$TARGET_QUANTIZED_DIR/" 2>/dev/null || true

log "--- Step 5: Final Inference Test ---"
python scripts/inference.py

log "--- Step 6: Upload to S3 ---"
if [ "$DEPLOY_ENV" == "runpod" ] || [ "$DEPLOY_ENV" == "cloud" ]; then
    MODELS_LOCAL_DIR="/workspace/models"
else
    MODELS_LOCAL_DIR="models"
fi
python scripts/upload_model.py --local-dir "$MODELS_LOCAL_DIR" --s3-prefix models

# Upload logs at the end of a successful run
upload_logs

log "--- Pipeline Complete! ---"

# Step 7: Terminate pod if running on RunPod
if [ "$DEPLOY_ENV" == "runpod" ] || [ "$DEPLOY_ENV" == "cloud" ]; then
    log "--- Step 7: Terminating Pod ---"
    if [ -f "./scripts/terminate_pod_remote.sh" ]; then
        bash ./scripts/terminate_pod_remote.sh || log "Warning: Pod termination script failed, but pipeline completed successfully."
    else
        log "Warning: terminate_pod.sh not found. Pod will continue running."
        log "You may need to terminate it manually via the RunPod console or CLI."
    fi
fi
