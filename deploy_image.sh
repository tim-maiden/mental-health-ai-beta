#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# --- CONFIGURATION ---
IMAGE_NAME="mental-health-ai"
DOCKER_USER="timarple"
TAG="latest"

echo "=============================================="
echo "   BUILDING & PUSHING DOCKER IMAGE (DOCKER HUB)"
echo "=============================================="

# 1. Build Image (Force AMD64 for Cloud Compatibility)
echo "--- Step 1: Building Image for linux/amd64 ---"
docker build --platform linux/amd64 -t ${IMAGE_NAME}:${TAG} .

# 2. Push to Docker Hub (For RunPod)
echo "--- Step 2: Processing for Docker Hub ---"
FULL_DOCKER_TAG="${DOCKER_USER}/${IMAGE_NAME}:${TAG}"
docker tag ${IMAGE_NAME}:${TAG} ${FULL_DOCKER_TAG}

echo "Pushing to Docker Hub: ${FULL_DOCKER_TAG}..."
docker push ${FULL_DOCKER_TAG}

echo "=============================================="
echo "   DEPLOYMENT ARTIFACT UPDATED SUCCESSFULLY"
echo "=============================================="
echo "Docker Hub: ${FULL_DOCKER_TAG}"
