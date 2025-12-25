#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# --- Helpers ---
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

error_handler() {
    log "Error occurred in script at line: $1"
    log "Pipeline failed."
    exit 1
}

trap 'error_handler $LINENO' ERR

# --- Defaults ---
CLEAN=true
DEPLOY_ENV="local"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
        pip install "optimum[onnxruntime]==1.18.0"
    else
        source venv/bin/activate
    fi
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

elif [ "$DEPLOY_ENV" == "runpod" ] || [ "$DEPLOY_ENV" == "cloud" ]; then
    # --- CLOUD (CUDA) ---
    log "Configuring Cloud Environment..."
    pip install --upgrade pip --break-system-packages
    pip install -r requirements.txt --break-system-packages
    log "Installing ONNX Runtime GPU..."
    pip install "optimum[onnxruntime-gpu]" --break-system-packages
    unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
else
    log "Error: Invalid deploy mode. Use 'local' or 'runpod'."
    exit 1
fi

# 3. Pipeline Steps

# Step 0: Data Snapshot (New for reproducibility)
log "--- Step 0: Data Ingestion (Snapshot) ---"
python scripts/ingest_data.py

log "--- Step 1: Statistical Audit (Risk Density) ---"
python scripts/audit_embeddings.py

log "--- Step 2: Dataset Compilation (Donut Strategy) ---"
python scripts/compile_dataset.py

log "--- Step 3: Train Classifier (ModernBERT/DeBERTa) ---"
python scripts/train_classifier.py

log "--- Step 4: Quantize Model (ONNX FP16) ---"
mkdir -p models/risk_classifier_quantized

# Verify optimum-cli exists
if ! command -v optimum-cli &> /dev/null; then
    log "Error: optimum-cli not found. Check installation."
    exit 1
fi

log "Exporting to ONNX (FP16)..."
optimum-cli export onnx --model models/risk_classifier_deberta_v1 --task text-classification --dtype fp16 models/risk_classifier_quantized/

# Rename for Inference Compatibility
if [ -f "models/risk_classifier_quantized/model.onnx" ]; then
    mv models/risk_classifier_quantized/model.onnx models/risk_classifier_quantized/model_quantized.onnx
fi

# Copy config files
log "Copying model artifacts..."
cp models/risk_classifier_deberta_v1/config.json models/risk_classifier_quantized/
cp models/risk_classifier_deberta_v1/tokenizer_config.json models/risk_classifier_quantized/
cp models/risk_classifier_deberta_v1/vocab.txt models/risk_classifier_quantized/ 2>/dev/null || true
cp models/risk_classifier_deberta_v1/tokenizer.json models/risk_classifier_quantized/ 2>/dev/null || true
cp models/risk_classifier_deberta_v1/special_tokens_map.json models/risk_classifier_quantized/ 2>/dev/null || true

log "--- Step 5: Final Inference Test ---"
python scripts/inference.py

log "--- Step 6: Upload to S3 ---"
python scripts/upload_model.py --local-dir models --s3-prefix models

log "--- Pipeline Complete! ---"
