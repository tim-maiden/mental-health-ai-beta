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
*   **`train_emotion_probe.py`**:
    *   Trains a Multi-Output Logistic Regression on the GoEmotions dataset.
    *   **Crucial Step**: Predicts sentiment for the `reddit_safe_embeddings` table in Supabase.
    *   *Why?* Allows `compile_dataset.py` to distinguish "Depressive Safe" from "Happy Safe."

*   **`ingest_data.py`**: Fetches embeddings from Supabase and creates a local snapshot (`data/raw_latest.pkl`).

*   **`compile_dataset.py`**: The central data processing script.
    *   Calculates Risk Density for every post.
    *   Filters out label noise.
    *   Performs Positive-Targeted Sampling for the Safe class.
    *   Generates k-NN Soft Labels for training.

#### 2. Model Training
*   **`train_classifier.py`**: Fine-tunes the DeBERTa model.
    *   **Hardware Aware**: Automatically selects `deberta-v3-small` for local testing and `deberta-v3-large` for cloud runs.
    *   Integrates with Weights & Biases for experiment tracking.

*   **`validate_gold.py`**: Runs evaluation metrics on the held-out Gold Test Set.

#### 3. Infrastructure
*   **`upload_datasets.py`**: Initial population of Supabase (Vectors + Metadata).

*   **`services/s3.py`**: Handles artifact versioning. Every run backups logs, configs, and weights to S3.

## Setup & Configuration

### 1. Prerequisites
*   Python 3.12+
*   Supabase Project (Vector Store)
*   OpenAI API Key (Embedding Generation)
*   Weights & Biases Account (Logging)
*   RunPod Account (for Cloud Training)

### 2. Environment Variables (`.env`)
```bash
# Core Credentials
OPENAI_KEY=sk-...
SUPABASE_URL=https://...
SUPABASE_SECRET=...

# Infrastructure (Optional but Recommended)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET_NAME=...
WANDB_API_KEY=...

# Deployment Controls
DEPLOY_ENV=local  # Set to 'runpod' or 'cloud' in production
```

### 3. First-Time Data Initialization
Before running the main pipeline, you must populate and enrich the database. This is a one-time operation.

```bash
# 1. Populate Supabase with Raw Data (Kaggle/HF)
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
