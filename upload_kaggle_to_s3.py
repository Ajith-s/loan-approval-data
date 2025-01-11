import os
import kaggle
import boto3
import tarfile


kaggle_username = os.getenv('KAGGLE_USERNAME')
kaggle_key = os.getenv('KAGGLE_KEY')

# Ensure that Kaggle credentials are properly set
if kaggle_username is None or kaggle_key is None:
    raise ValueError("Kaggle API credentials are not set. Please check your .zshrc file.")

# Set up Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = kaggle_username
os.environ['KAGGLE_KEY'] = kaggle_key

dataset_name = "taweilo/loan-approval-classification-data"
# Download the dataset from Kaggle
try:
    print(f"Downloading dataset: {dataset_name}...")
    kaggle.api.dataset_download_files(dataset_name, path='.', unzip=True)
    print("Download successful!")
except Exception as e:
    print(f"Error downloading dataset: {e}")

# Upload to S3
s3 = boto3.client("s3")
bucket_name = "loan-data-kaggle-ajiths"
local_file = "./data/loan_data.csv"
model_file_path = "loan_approval_model.pkl"
s3_model_path = "models/loan_approval_model.pkl"
tar_gz_path = "loan_approval_model.tar.gz"


s3.upload_file(local_file, bucket_name, "s3_loan_data.csv")
print("Data uploaded to S3 successfully!")

s3.upload_file(model_file_path, bucket_name, s3_model_path)
with tarfile.open(tar_gz_path, "w:gz") as tar:
    tar.add(model_file_path, arcname=os.path.basename(model_file_path))

s3.upload_file(tar_gz_path, "loan-data-kaggle-ajiths", "models/loan_approval_model.tar.gz")
print("Model uploaded to S3 successfully.")
