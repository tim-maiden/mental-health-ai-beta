#!/bin/bash
# Helper script to terminate a RunPod pod from your local machine
# This requires runpodctl to be authorized on your local machine

set -e

POD_NAME="${1:-mental-health-ai-training}"

echo "=============================================="
echo "   TERMINATING RUNPOD POD"
echo "=============================================="

# Check if runpodctl is available
if ! command -v runpodctl &> /dev/null; then
    echo "Error: runpodctl not found."
    echo ""
    echo "To install runpodctl:"
    echo "  pip install runpod"
    echo ""
    echo "To authorize runpodctl:"
    echo "  runpodctl config"
    echo ""
    echo "Alternatively, you can terminate the pod via:"
    echo "  1. RunPod web console: https://www.runpod.io/console/pods"
    echo "  2. The pod will auto-terminate after the pipeline completes"
    exit 1
fi

# Check if runpodctl is authorized
if ! runpodctl pods list &> /dev/null; then
    echo "Error: runpodctl is not authorized."
    echo ""
    echo "To authorize runpodctl, run:"
    echo "  runpodctl config"
    echo ""
    echo "This will prompt you for your RunPod API key."
    echo "You can find your API key at: https://www.runpod.io/console/user/settings"
    exit 1
fi

echo "Looking for pod: $POD_NAME"
echo ""

# Get pod ID by name
POD_ID=$(runpodctl pods list --output json 2>/dev/null | \
    python3 -c "import sys, json; \
    pods = json.load(sys.stdin); \
    pod = next((p for p in pods if p.get('name') == '$POD_NAME'), None); \
    print(pod['id'] if pod else '')" 2>/dev/null || echo "")

if [ -z "$POD_ID" ]; then
    echo "Error: Could not find pod with name '$POD_NAME'"
    echo ""
    echo "Available pods:"
    runpodctl pods list
    exit 1
fi

echo "Found pod ID: $POD_ID"
echo "Terminating pod..."
echo ""

if runpodctl remove pod "$POD_ID"; then
    echo ""
    echo "=============================================="
    echo "   POD TERMINATED SUCCESSFULLY"
    echo "=============================================="
else
    echo ""
    echo "Error: Failed to terminate pod."
    exit 1
fi

