# Deployment Scripts

This directory contains scripts for managing RunPod infrastructure and deployments.

## Scripts

### `create_pod.sh`
Creates a new RunPod pod with the specified configuration.
- **Usage**: `./deploy/create_pod.sh`
- **Requires**: `runpodctl` CLI authorized on your local machine
- **Purpose**: Spin up a new pod for training/inference

### `deploy_image.sh`
Builds and pushes the Docker image to Docker Hub.
- **Usage**: `./deploy/deploy_image.sh`
- **Requires**: Docker installed and logged into Docker Hub
- **Purpose**: Update the container image that pods will use

### `terminate_pod_local.sh`
Manually terminates a RunPod pod from your local machine.
- **Usage**: `./deploy/terminate_pod_local.sh [pod-name]`
- **Requires**: `runpodctl` CLI authorized on your local machine
- **Purpose**: Emergency termination or cleanup when pod doesn't auto-terminate

## Workflow

1. **Build & Push Image**: `./deploy/deploy_image.sh`
2. **Create Pod**: `./deploy/create_pod.sh`
3. **Pod Auto-Terminates**: The pod will automatically terminate itself after `run.sh` completes
4. **Manual Termination** (if needed): `./deploy/terminate_pod_local.sh`

## Notes

- The pod will automatically terminate itself after the pipeline completes (see `scripts/terminate_pod_remote.sh`)
- You only need to use `terminate_pod_local.sh` if something goes wrong or you want to stop the pod early
- All scripts in this directory are meant to be run from your **local machine**, not inside the pod

