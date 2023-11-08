import tarfile
import os

tar_gz_path = './dataset/dataset.tar.gz'
target_directory = './dataset'

# Function to extract the tar.gz file
def extract_tar_gz(tar_gz_path, target_directory):
    # Open the tar.gz file
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        # Extract its contents into the target directory
        tar.extractall(path=target_directory)
    print(f"Files extracted to {target_directory}")


