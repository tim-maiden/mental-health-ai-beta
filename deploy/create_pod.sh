#!/bin/bash
set -e

# --- CONFIGURATION ---
POD_NAME="mental-health-ai-training"
GPU_TYPE="NVIDIA RTX 6000 Ada Generation" 
GPU_COUNT=1
IMAGE_NAME="timarple/mental-health-ai:latest"
CONTAINER_DISK_SIZE=50
VOLUME_SIZE=50
ENV_FILE=".env"

# --- CONSTRUCT COMMAND ARRAY ---
# Using an array avoids issues with spaces in arguments (like GPU_TYPE)
CMD=(runpodctl create pod)
CMD+=(--name "$POD_NAME")
CMD+=(--gpuType "$GPU_TYPE")
CMD+=(--gpuCount "$GPU_COUNT")
CMD+=(--secureCloud)
CMD+=(--imageName "$IMAGE_NAME")
CMD+=(--containerDiskSize "$CONTAINER_DISK_SIZE")
CMD+=(--volumeSize "$VOLUME_SIZE")
CMD+=(--ports "8888/http")

# --- LOAD ENVIRONMENT VARIABLES ---
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
        fi
    done < "$ENV_FILE"
else
    echo "Warning: $ENV_FILE not found. No environment variables will be set."
fi

# Ensure AWS vars are passed even if not in .env (if exported in shell)
if [ -n "$AWS_ACCESS_KEY_ID" ]; then CMD+=(--env "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID"); fi
if [ -n "$AWS_SECRET_ACCESS_KEY" ]; then CMD+=(--env "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY"); fi
if [ -n "$AWS_REGION" ]; then CMD+=(--env "AWS_REGION=$AWS_REGION"); fi
if [ -n "$S3_BUCKET_NAME" ]; then CMD+=(--env "S3_BUCKET_NAME=$S3_BUCKET_NAME"); fi

# --- CREATE POD ---
echo "Creating Pod on RunPod..."
echo "Pod Name: $POD_NAME"
echo "GPU: $GPU_TYPE x $GPU_COUNT"
echo "Image: $IMAGE_NAME"

# Execute the command
"${CMD[@]}"

echo "Pod creation command sent."
