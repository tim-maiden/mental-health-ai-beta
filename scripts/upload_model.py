import sys
import os
import argparse
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.services.s3 import upload_directory_to_s3

def main():
    parser = argparse.ArgumentParser(description="Upload models directory to S3.")
    parser.add_argument("--local-dir", type=str, required=True, help="Local directory to upload")
    parser.add_argument("--s3-prefix", type=str, default="models", help="S3 prefix (folder)")
    
    args = parser.parse_args()
    
    upload_directory_to_s3(args.local_dir, args.s3_prefix)

if __name__ == "__main__":
    main()

