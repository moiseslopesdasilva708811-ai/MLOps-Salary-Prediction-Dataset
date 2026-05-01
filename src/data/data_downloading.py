import kagglehub
import os
import shutil
import pandas as pd

# destination folder path
destination_folder = 'dataset'
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    print(f"Folder '{destination_folder}' created!")

# Download the latest version of the dataset
# kagglehub downloads files to a temporary system cache location
path = kagglehub.dataset_download("ayessa/salary-prediction-classification")

print("Files downloaded to:", path)

# Move the specific file to your local 'data' folder
# We use 'salary.csv' as per the dataset's structure
source_file = os.path.join(path, "salary.csv")
destination_file = os.path.join(destination_folder, "salary.csv")

if os.path.exists(source_file):
    shutil.copy(source_file, destination_file)
    print(f"✅ Success! The file was copied to: {destination_file}")
else:
    # If the filename is different, this lists available files for debugging
    print("File 'salary.csv' not found. Available files:")
    print(os.listdir(path))

# Load the dataset directly from your local folder
df = pd.read_csv(destination_file)
print("\nFirst records of your local CSV:")
print(df.head())