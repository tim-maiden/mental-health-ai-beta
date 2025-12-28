#!/bin/bash
# Script to download the most recent log file from S3

echo "Finding latest log..."
LATEST_LOG=$(aws s3 ls s3://mhai-model-weights/logs/ --recursive | sort | tail -n 1 | awk '{print $4}')

if [ -z "$LATEST_LOG" ]; then
    echo "No logs found in s3://mhai-model-weights/logs/"
    exit 1
fi

echo "Downloading $LATEST_LOG..."
mkdir -p logs
aws s3 cp "s3://mhai-model-weights/$LATEST_LOG" ./logs/latest_remote.log

echo "Downloaded to logs/latest_remote.log"

