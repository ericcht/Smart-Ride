import kagglehub
import os
import pandas as pd
import shutil
from pathlib import Path

def download_uber_dataset():
    print("=" * 60)
    print("DOWNLOADING UBER DATASET")
    print("=" * 60)
    
    try:
        # Download the dataset
        print("Downloading Uber Ride Analytics dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download("yashdevladdha/uber-ride-analytics-dashboard")
        
        print(f"Dataset downloaded to: {dataset_path}")
        
        # List all files in the dataset
        print("\nFiles in the dataset:")
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  - {file} ({file_size:.1f} MB)")
        
        # Find CSV files
        csv_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if csv_files:
            print(f"\nFound {len(csv_files)} CSV files:")
            for csv_file in csv_files:
                print(f"  - {csv_file}")
            
            # Copy the main dataset file to our data/raw directory
            main_csv = csv_files[0]  # Use the first CSV file
            target_path = "data/raw/uber_data.csv"
            
            # Create directory if it doesn't exist
            os.makedirs("data/raw", exist_ok=True)
            
            # Copy the file
            shutil.copy2(main_csv, target_path)
            print(f"\nMain dataset copied to: {target_path}")
            
            # Explore the dataset structure
            print("\nExploring dataset structure...")
            df = pd.read_csv(target_path)
            
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst few rows:")
            print(df.head())
            
            print(f"\nData types:")
            print(df.dtypes)
            
            print(f"\nMissing values:")
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(missing[missing > 0])
            else:
                print("No missing values!")
            
            return True
            
        else:
            print("No CSV files found in the dataset")
            return False
            
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False

def main():
    success = download_uber_dataset()
    
    if success:
        print("\nDataset downloaded successfully! Data is in data/raw/uber_data.csv")
    else:
        print("\nFailed to download dataset")

if __name__ == "__main__":
    main()
