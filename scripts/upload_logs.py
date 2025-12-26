import sys
import os
import argparse
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.services.s3 import upload_file_to_s3

def main():
    parser = argparse.ArgumentParser(description="Upload log file to S3.")
    parser.add_argument("--log-file", type=str, required=True, help="Path to the log file")
    parser.add_argument("--s3-prefix", type=str, default="logs", help="S3 prefix (folder)")
    
    args = parser.parse_args()
    
    upload_file_to_s3(args.log_file, args.s3_prefix)

if __name__ == "__main__":
    main()

