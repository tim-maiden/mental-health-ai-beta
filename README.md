# AI Mental Health - Reddit Risk Classifier

This project implements a comprehensive pipeline to train a mental health risk classifier using Reddit data. It includes data ingestion, processing, embedding generation, model training, distillation, and deployment-ready inference.

## Project Structure

### Core Workflow
The main pipeline is orchestrated by `run.sh`. This script handles the end-to-end process:
1.  **Environment Setup**: Handles local (venv) vs. cloud (RunPod) configuration.
2.  **Data Ingestion**: Fetches raw data from Supabase.
3.  **Dataset Compilation**: Prepares training data (teacher training).
4.  **Teacher Training**: Trains a DeBERTa-based classifier.
5.  **Inference & Distillation**: (Optional) Generates silver labels and trains a smaller student model.

### Key Scripts (`scripts/`)

#### 1. Data Management
*   **`upload_datasets.py`**: The entry point for populating the database. It handles downloading raw datasets (Kaggle/HuggingFace), chunking text, generating embeddings (OpenAI), and uploading to Supabase.
    *   Flags: `--lmsys`, `--wildchat`, `--mental-health`, `--controls`, `--goemotions`
*   **`ingest_data.py`**: Fetches the processed embeddings from Supabase and creates a local snapshot (`data/raw_latest.pkl`) for training. This is step 1 of the main pipeline.
*   **`compile_dataset.py`**: Takes the raw snapshot and compiles it into training/testing splits (`final_train.jsonl`, `test.jsonl`), optionally using k-NN for soft labeling.

#### 2. Model Training
*   **`train_classifier.py`**: Trains the "Teacher" model (DeBERTa Large/Small). It supports multi-label classification and integrates with Weights & Biases for logging.
*   **`train_distilled.py`**: (Experimental) Trains a smaller "Student" model (e.g., DistilBERT) using the Teacher's outputs.
*   **`train_emotion_probe.py`**: A standalone utility to train a lightweight linear probe on GoEmotions embeddings. It predicts emotions for the "Safe" Reddit control data to enable emotion-balanced sampling.
    *   Usage: `python scripts/train_emotion_probe.py --train --predict`

#### 3. Inference & Ops
*   **`inference.py`**: Runs the trained model on new text or datasets (like WildChat) to generate predictions or evaluate performance.
*   **`upload_model.py` / `upload_logs.py`**: Utilities to backup artifacts and logs to AWS S3.
*   **`download_datasets.py`**: Fetches pre-compiled datasets from S3 to speed up cloud training.

#### 4. Deployment Helpers (`deploy/`)
*   **`create_pod.sh`**: Helper to spin up a RunPod GPU instance.
*   **`deploy_image.sh`**: Builds and pushes the Docker image.

## Setup & Usage

### 1. Prerequisites
*   Python 3.12+
*   Supabase Account (for data storage)
*   OpenAI API Key (for embeddings)
*   AWS S3 (optional, for artifact storage)

### 2. Environment Variables
Create a `.env` file with the following:
```bash
OPENAI_KEY=sk-...
SUPABASE_URL=https://...
SUPABASE_SECRET=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET_NAME=...
WANDB_API_KEY=...
```

### 3. Data Enrichment (Pre-requisite)
To enable emotion-driven sampling, you must first enrich the safety dataset with emotion labels. This is a **one-time setup** (or whenever you add new raw data):
```bash
# 1. Train the probe on GoEmotions
# 2. Predict labels for the Reddit Safety dataset in Supabase
python scripts/train_emotion_probe.py --train --predict
```

### 4. Running the Pipeline
Once data is enriched, run the main pipeline. This will ingest the enriched data, compile the dataset (using emotions for balancing), and train the model.

**Local Development:**
```bash
./run.sh --deploy local
```

**Cloud (RunPod):**
```bash
./run.sh --deploy runpod
```

## Data Flow
1.  **Raw Sources** (Kaggle/HF) -> `upload_datasets.py` -> **Supabase** (Embeddings)
2.  **Supabase** -> `ingest_data.py` -> **Local Snapshot** (`.pkl`)
3.  **Local Snapshot** -> `compile_dataset.py` -> **Training Files** (`.jsonl`)
4.  **Training Files** -> `train_classifier.py` -> **Model Artifacts**

