import os
import sys
import boto3
import argparse
from botocore.exceptions import NoCredentialsError, ClientError

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    DATA_DIR,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    S3_BUCKET_NAME
)

def get_s3_client():
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("Error: AWS credentials missing.")
        return None
    
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

def download_file_from_s3(s3_key, local_path):
    s3 = get_s3_client()
    if not s3:
        return False

    try:
        # Check if file exists first
        try:
            s3.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        except ClientError:
            print(f"File not found in S3: {s3_key}")
            return False

        print(f"Downloading s3://{S3_BUCKET_NAME}/{s3_key} -> {local_path}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
        return True
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except Exception as e:
        print(f"Failed to download {s3_key}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download compiled datasets from S3.")
    parser.add_argument("--s3-prefix", type=str, default="data/latest", help="S3 prefix folder to download from")
    parser.add_argument("--force", action="store_true", help="Overwrite local files if they exist")
    args = parser.parse_args()

    print(f"--- Downloading Compiled Datasets from S3 (Prefix: {args.s3_prefix}) ---")
    
    # Define files to download
    # We construct the S3 key by joining the prefix with the filename
    files_map = {
        os.path.basename(TRAIN_FILE): TRAIN_FILE,
        os.path.basename(TEST_FILE): TEST_FILE,
        "subreddit_mapping.json": os.path.join(DATA_DIR, "subreddit_mapping.json")
    }

    success = True
    downloaded_count = 0

    for filename, local_path in files_map.items():
        if os.path.exists(local_path) and not args.force:
            print(f"File {local_path} already exists. Skipping.")
            continue
            
        # The S3 key is prefix + / + filename
        # Ensure prefix doesn't have trailing slash for join, but handle it if it does
        prefix = args.s3_prefix.rstrip("/")
        s3_key = f"{prefix}/{filename}"
        
        if not download_file_from_s3(s3_key, local_path):
            print(f"Failed to download {filename}")
            success = False
        else:
            downloaded_count += 1

    if success:
        print(f"Successfully downloaded/verified all {len(files_map)} files.")
        sys.exit(0)
    else:
        print("One or more files failed to download.")
        sys.exit(1)

if __name__ == "__main__":
    main()
