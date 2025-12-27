# Use RunPod's official PyTorch base image (includes CUDA, Python, PyTorch)
# FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Set the working directory
WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEPLOY_ENV=cloud

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
# 1. Standard requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the run script is executable
RUN chmod +x run.sh

# Default command to run when the pod starts
# You can override this command in the RunPod UI if you just want to open a terminal
CMD ["./run.sh", "--deploy", "runpod"]

