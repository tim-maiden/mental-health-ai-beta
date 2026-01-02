# AI Mental Health - Risk Classification Pipeline

This project implements a high-precision pipeline to classify mental health risk (Suicide/Depression vs. Safe) using Reddit data. Unlike standard classifiers, this system uses **Embedding-Based Density Filtering** and **Emotion-Guided Sampling** to curate a training set that distinguishes *clinical risk* from *safe emotional expression*.

## Core Architecture

The pipeline consists of three distinct phases:

1.  **Data Curation (The "Teacher" Data):**
    *   **Ingestion:** Embeds raw posts using OpenAI `text-embedding-3`.
    *   **Emotion Probing:** A specialized Logistic Regression probe labels the "Safe" control set with sentiments (Positive, Negative, Ambiguous).
    *   **Compilation:**
        *   **Density Filtering:** Prunes "Risk" posts that are semantically distinct from the risk cluster (outliers).
        *   **Hard Negative Mining:** Identifies "Safe" posts that look like Risk (e.g., venting) to improve decision boundaries.
        *   **Positive Injection:** Explicitly oversamples *Positive* Safe posts (Joy, Pride) to prevent the model from learning "Sad = Risk."

2.  **Teacher Training:**
    *   Fine-tunes **DeBERTa-v3** (Large on Cloud, Small on Local) on the curated dataset.
    *   Uses **Soft Labels** generated via weighted k-NN from the embedding space to smooth the loss landscape.

3.  **Inference & Distillation (Optional):**
    *   Generates "Silver Labels" for large unlabeled datasets (e.g., WildChat).
    *   Trains lightweight Student models (DistilBERT/MobileBERT) for edge deployment.

---

## Project Structure

```
.
├── data/                       # Local data cache
│   ├── goemotions_cache.pkl
│   └── reddit_control_cache.parquet
├── deploy/                     # Deployment utilities
│   ├── create_pod.sh           # Script to launch RunPod instance
│   ├── deploy_image.sh         # Builds and pushes Docker image
│   ├── terminate_pod_local.sh  # Terminates pod from local machine
│   └── README.md
├── logs/                       # Execution logs
├── models/                     # Trained models and artifacts
├── scripts/                    # Pipeline executables
│   ├── cleanup.sh              # Removes temporary artifacts
│   ├── compile_dataset.py      # Core data processing & soft labeling
│   ├── download_datasets.py    # Fetches compiled data from S3
│   ├── download_latest_log.sh  # Retrieves logs from S3
│   ├── inference.py            # Model inference (Test/WildChat)
│   ├── ingest_data.py          # Data snapshot creation
│   ├── terminate_pod_remote.sh # Self-termination script for pods
│   ├── train_classifier.py     # Teacher model training (DeBERTa)
│   ├── train_distilled.py      # Student model training (DistilBERT)
│   ├── train_emotion_probe.py  # Trains the emotion logistic regression
│   ├── upload_datasets.py      # Initial Supabase population
│   ├── upload_logs.py          # Log archival to S3
│   ├── upload_model.py         # Model artifact archival to S3
│   └── validate_gold.py        # Gold set evaluation
├── src/                        # Source code library
│   ├── analysis/               # Metrics and evaluation logic
│   ├── core/                   # Clients (Supabase, OpenAI) and utils
│   ├── data/                   # Data loaders and storage handlers
│   ├── modeling/               # Training loops and inference logic
│   ├── services/               # Services (S3, Embeddings, RateLimiter)
│   ├── supabase_setup/         # SQL schemas for database
│   └── config.py               # Central configuration
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies (Cloud)
├── requirements_local.txt      # Python dependencies (Local/Mac)
└── run.sh                      # Main pipeline entry point
```

### Main Entry Point: `run.sh`
Orchestrates the end-to-end pipeline. It automatically detects the environment (`local` vs `runpod`) and adjusts compute resources accordingly.

```bash
# Local Development (Mac/MPS - Uses DeBERTa-Small)
./run.sh --deploy local

# Cloud Production (H100/CUDA - Uses DeBERTa-Large)
./run.sh --deploy runpod
```

### Key Scripts (`scripts/`)

#### 1. Data Engineering
*   **`train_emotion_probe.py`**: Trains a Multi-Output Logistic Regression on the GoEmotions dataset to predict sentiment for `reddit_safe_embeddings`. Crucial for distinguishing "Depressive Safe" from "Happy Safe."
*   **`ingest_data.py`**: Fetches embeddings from Supabase and creates a local snapshot (`raw_latest.pkl`/parquet).
*   **`compile_dataset.py`**: The central data processor. Calculates Risk Density, performs Positive-Targeted Sampling, and generates k-NN Soft Labels.
*   **`download_datasets.py`**: Checks S3 for pre-compiled data to avoid re-running ingestion/compilation on every run.

#### 2. Model Training
*   **`train_classifier.py`**: Fine-tunes the DeBERTa model. Automatically selects `deberta-v3-small` (Local) or `deberta-v3-large` (Cloud).
*   **`train_distilled.py`**: Trains a smaller Student model (e.g., DistilBERT) using "Silver Labels" generated by the Teacher model.
*   **`validate_gold.py`**: Runs evaluation metrics on the held-out Gold Test Set.

#### 3. Inference & Ops
*   **`inference.py`**: Runs the trained model on test inputs, LMSYS data, or WildChat data (from Supabase) to generate predictions or silver labels.
*   **`upload_*.py`**: Utilities for versioning datasets, models, and logs to S3.

## Setup & Configuration

### 1. Prerequisites
*   Python 3.12+
*   Supabase Project (Vector Store)
*   OpenAI API Key (Embedding Generation)
*   Weights & Biases Account (Logging)
*   RunPod Account (for Cloud Training - Optional)
*   AWS S3 (for Artifact Storage - Optional)

### 2. Environment Variables (`.env`)
Create a `.env` file in the root directory:

```bash
# Core Credentials
OPENAI_KEY=sk-...
SUPABASE_URL=https://...
SUPABASE_SECRET=...

# Infrastructure (Optional but Recommended)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-west-2
S3_BUCKET_NAME=...
WANDB_API_KEY=...

# Deployment Controls
DEPLOY_ENV=local  # Set to 'runpod' or 'cloud' in production
```

### 3. First-Time Data Initialization
Before running the main pipeline, you must populate and enrich the database. This is a one-time operation.

**A. Database Schema Setup:**
Before running any Python scripts, you must create the necessary tables in Supabase.
1.  Navigate to your Supabase SQL Editor.
2.  Run the SQL scripts located in `src/supabase_setup/` in the following order:
    *   `setup_goemotions_table.sql`
    *   `setup_reddit_table.sql`
    *   `setup_safe_control_table.sql`
    *   `setup_wildchat_table.sql` (Optional, for silver labeling)
    *   `add_emotion_scores.sql`

**B. Data Population:**
```bash
# 1. Populate Supabase with Raw Data (Kaggle/HF)
# Flags available: --mental-health, --controls, --goemotions, --lmsys, --wildchat
python scripts/upload_datasets.py --mental-health --controls --goemotions

# 2. Enrich 'Safe' data with Emotion Labels (CRITICAL)
# Without this, the compiler defaults to random sampling.
python scripts/train_emotion_probe.py --train --predict
```

## Technical Details

### Density Calculation
We rely on the hypothesis that Risk is a Manifold. We calculate the average cosine similarity of a post to its *k* nearest neighbors in the "Teacher" set.
*   **High Density Risk** = High Confidence Risk.
*   **High Density Safe + High Similarity to Risk** = Hard Negative (Ambiguous).

### Hardware Logic (`src/config.py`)
The pipeline prevents OOM errors by checking `DEPLOY_ENV`:
*   **Local**: `batch_size=16`, `model=deberta-v3-small`, `fp16=False` (MPS compatibility).
*   **RunPod**: `batch_size=32+`, `model=deberta-v3-large`, `fp16=True` (A100/H100 optimization).

### Emotion Targeting
To avoid the "Neutrality Trap" (where the model learns that Sadness is inherently Risk), we use the Emotion Probe to force-inject Positive Sentiment (Joy, Optimism, Pride) into the Safe training set. This ensures the model learns to distinguish Clinical Depression from Normal Sadness.
