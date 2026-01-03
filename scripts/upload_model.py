import sys
import os
import argparse
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload models directory to Hugging Face Hub.")
    parser.add_argument("--local-dir", type=str, required=True, help="Local directory to upload")
    parser.add_argument("--repo-id", type=str, required=True, help="Target HF Repo ID (e.g. username/model-name)")
    parser.add_argument("--path-in-repo", type=str, default=".", help="Path within the repo")
    
    args = parser.parse_args()
    
    print(f"--- Uploading {args.local_dir} to HF Hub: {args.repo_id} ---")
    
    token = os.getenv("HF_TOKEN")
    if token:
        print(f"HF_TOKEN detected: {token[:4]}...{token[-4:]}")
    else:
        print("WARNING: HF_TOKEN not found in environment variables! Upload may fail if not cached.")

    api = HfApi(token=token)
    
    # Create repo if it doesn't exist (private by default to be safe)
    api.create_repo(repo_id=args.repo_id, exist_ok=True, private=True)
    
    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
        ignore_patterns=["checkpoint-*"] # Skip intermediate checkpoints
    )
    print("Upload complete.")

if __name__ == "__main__":
    main()
