#!/bin/bash
# Script to terminate the RunPod pod after pipeline completion
# This uses the RunPod API via runpodctl which is pre-installed in pods

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Check if we're running in a RunPod environment
if [ -z "$RUNPOD_POD_ID" ]; then
    log "Not running in RunPod environment (RUNPOD_POD_ID not set). Skipping pod termination."
    exit 0
fi

# Check if runpodctl is available
if ! command -v runpodctl &> /dev/null; then
    log "Warning: runpodctl not found. Cannot terminate pod automatically."
    log "You may need to terminate the pod manually via the RunPod console."
    exit 0
fi

log "Terminating RunPod pod: $RUNPOD_POD_ID"
log "This will permanently delete the pod and all non-persistent data."

# Terminate the pod
if runpodctl remove pod "$RUNPOD_POD_ID"; then
    log "Pod termination command sent successfully."
    log "The pod will shut down shortly."
else
    log "Warning: Failed to terminate pod. You may need to terminate it manually."
    exit 1
fi

