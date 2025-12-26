import os
import boto3
from botocore.exceptions import NoCredentialsError
from src.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME

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

def upload_directory_to_s3(local_dir, s3_prefix="models"):
    """
    Recursively uploads a directory to S3.
    """
    s3 = get_s3_client()
    if not s3:
        return False
    
    if not S3_BUCKET_NAME:
        print("Error: S3_BUCKET_NAME not set.")
        return False

    if not os.path.isdir(local_dir):
        print(f"Error: Local directory {local_dir} does not exist.")
        return False

    print(f"--- Uploading {local_dir} to s3://{S3_BUCKET_NAME}/{s3_prefix} ---")
    
    for root, dirs, files in os.walk(local_dir):
        # Exclude checkpoint directories to speed up upload
        dirs[:] = [d for d in dirs if not d.startswith('checkpoint-')]

        for file in files:
            local_path = os.path.join(root, file)
            
            # Calculate S3 path
            relative_path = os.path.relpath(local_path, local_dir)
            s3_path = os.path.join(s3_prefix, relative_path)
            
            try:
                print(f"Uploading {file} -> {s3_path}...")
                s3.upload_file(local_path, S3_BUCKET_NAME, s3_path)
            except FileNotFoundError:
                print(f"The file was not found: {local_path}")
            except NoCredentialsError:
                print("Credentials not available")
                return False
            except Exception as e:
                print(f"Failed to upload {file}: {e}")
                return False
                
    print("Upload complete.")
    return True

def upload_file_to_s3(local_path, s3_prefix="logs"):
    """
    Uploads a single file to S3.
    """
    s3 = get_s3_client()
    if not s3:
        return False
    
    if not S3_BUCKET_NAME:
        print("Error: S3_BUCKET_NAME not set.")
        return False

    if not os.path.isfile(local_path):
        print(f"Error: Local file {local_path} does not exist.")
        return False
        
    filename = os.path.basename(local_path)
    s3_path = os.path.join(s3_prefix, filename)
    
    try:
        print(f"--- Uploading {local_path} to s3://{S3_BUCKET_NAME}/{s3_path} ---")
        s3.upload_file(local_path, S3_BUCKET_NAME, s3_path)
        print("Upload complete.")
        return True
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except Exception as e:
        print(f"Failed to upload {local_path}: {e}")
        return False

