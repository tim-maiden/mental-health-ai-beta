#!/bin/bash
set -e

# --- CONFIGURATION ---
POD_NAME="mental-health-ai-distillation"
GPU_TYPE="NVIDIA RTX 6000 Ada Generation" # Cheaper than A100, sufficient for DistilBERT
GPU_COUNT=1
IMAGE_NAME="timarple/mental-health-ai:latest"
CONTAINER_DISK_SIZE=40
VOLUME_SIZE=40
ENV_FILE=".env"
DOCKER_CMD="bash scripts/run_distillation_job.sh"

# --- CONSTRUCT COMMAND ARRAY ---
# Using an array avoids issues with spaces in arguments (like GPU_TYPE)
CMD=(runpodctl create pod)
CMD+=(--name "$POD_NAME")
CMD+=(--gpuType "$GPU_TYPE")
CMD+=(--gpuCount "$GPU_COUNT")
CMD+=(--imageName "$IMAGE_NAME")
CMD+=(--containerDiskSize "$CONTAINER_DISK_SIZE")
CMD+=(--volumeSize "$VOLUME_SIZE")
CMD+=(--env "DEPLOY_ENV=runpod")
CMD+=(--args "$DOCKER_CMD")

# --- LOAD ENVIRONMENT VARIABLES ---
HF_TOKEN_FOUND=false

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE..."
    while IFS='=' read -r key value || [ -n "$key" ]; do
        # Skip comments and empty lines
        [[ $key =~ ^#.* ]] && continue
        [[ -z $key ]] && continue
        
        # Trim whitespace
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        
        # Remove surrounding quotes if present in .env
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        
        if [ -n "$key" ] && [ -n "$value" ]; then
            CMD+=(--env "$key=$value")
            if [ "$key" == "HF_TOKEN" ]; then
                HF_TOKEN_FOUND=true
            fi
        fi
    done < "$ENV_FILE"
else
    echo "Warning: $ENV_FILE not found. No environment variables will be set."
fi

if [ "$HF_TOKEN_FOUND" = false ]; then
    echo "⚠️  WARNING: HF_TOKEN not found in .env file!"
    echo "   The pipeline will FAIL to upload the model to Hugging Face."
    echo "   Please add HF_TOKEN=your_token to .env before continuing."
    read -p "   Press ENTER to continue anyway (or Ctrl+C to abort)..."
fi

# Ensure AWS vars are passed even if not in .env (if exported in shell)
if [ -n "$AWS_ACCESS_KEY_ID" ]; then CMD+=(--env "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID"); fi
if [ -n "$AWS_SECRET_ACCESS_KEY" ]; then CMD+=(--env "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY"); fi
if [ -n "$AWS_REGION" ]; then CMD+=(--env "AWS_REGION=$AWS_REGION"); fi
if [ -n "$S3_BUCKET_NAME" ]; then CMD+=(--env "S3_BUCKET_NAME=$S3_BUCKET_NAME"); fi

# --- CREATE POD ---
echo "Launching Distillation Pod..."
echo "Pod Name: $POD_NAME"
echo "GPU: $GPU_TYPE x $GPU_COUNT"
echo "Image: $IMAGE_NAME"
echo "Command: $DOCKER_CMD"

# Execute the command
"${CMD[@]}"

echo "Pod launch request sent."
