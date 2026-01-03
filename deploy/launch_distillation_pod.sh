#!/bin/bash
set -e

# --- CONFIGURATION ---
POD_NAME="mental-health-ai-distillation"
GPU_TYPE="NVIDIA RTX 6000 Ada Generation" # Cheaper than A100, sufficient for DistilBERT
GPU_COUNT=1
IMAGE_NAME="timarple/mental-health-ai:latest"
ENV_FILE=".env"

# --- COMMAND ---
# We override the Docker CMD to run our specific job script
DOCKER_CMD="bash scripts/run_distillation_job.sh"

echo "Launching Distillation Pod..."

# Construct RunPod command
# Note: We pass the command as arguments to runpodctl create pod
runpodctl create pod \
    --name "$POD_NAME" \
    --gpuType "$GPU_TYPE" \
    --gpuCount "$GPU_COUNT" \
    --imageName "$IMAGE_NAME" \
    --containerDiskSize 40 \
    --volumeSize 40 \
    --env "HF_TOKEN=$(grep HF_TOKEN $ENV_FILE | cut -d '=' -f2)" \
    --env "WANDB_API_KEY=$(grep WANDB_API_KEY $ENV_FILE | cut -d '=' -f2)" \
    --env "DEPLOY_ENV=runpod" \
    --dockerArgs "$DOCKER_CMD"

echo "Pod launch request sent."

