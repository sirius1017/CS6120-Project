import os
import json
import requests
import sys
from pathlib import Path
import argparse
import gdown  # For downloading from Google Drive
import subprocess

def download_dataset(url=None, output_dir=None, use_gdrive=True):
    """
    Download and process the recipe dataset.
    
    Args:
        url: URL to download the dataset from. If None, will use default URL.
        output_dir: Directory to save the dataset. If None, will use 'data' directory.
        use_gdrive: Whether to use Google Drive for downloading. Default is True.
    """
    print("Downloading recipe dataset...")
    
    # Create data directory if it doesn't exist
    data_dir = Path(output_dir) if output_dir else Path(".")
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / "recipes.json"
    
    print(f"Dataset will be saved to: {output_path}")
    print(f"Note: The dataset is approximately 200MB. This may take some time to download.")
    
    try:
        # Check if file already exists
        if output_path.exists():
            print(f"Dataset already exists at {output_path}. Skipping download.")
            return
        
        if use_gdrive:
            #gdrive_url = "https://drive.google.com/file/d/1Ts5iK3wcYWloyRmYk828p7g_6rU7XI88/view?usp=drive_link"
            print("Downloading from Google Drive...")
            #gdown.download(gdrive_url, str(output_path), quiet=False)
            gdown.download(id="1Ts5iK3wcYWloyRmYk828p7g_6rU7XI88", output=str(output_path), quiet=False)

            
            if output_path.exists():
                print(f"Dataset downloaded successfully to {output_path}")
            else:
                raise Exception("Download failed. File not found after download.")
       
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please manually download the dataset and place it in the project directory.")

def run_preprocessing():
    """
    Run the data preprocessing script after downloading the dataset.
    """
    print("\nRunning data preprocessing...")
    try:
        # Run the data_preprocessing.py script
        subprocess.run([sys.executable, "data_preprocessing.py"], check=True)
        print("Data preprocessing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during data preprocessing: {e}")
    except Exception as e:
        print(f"Unexpected error during data preprocessing: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download and process the recipe dataset')
    parser.add_argument('--url', help='URL to download the dataset from')
    parser.add_argument('--output-dir', help='Directory to save the dataset')
    parser.add_argument('--no-gdrive', action='store_true', help='Do not use Google Drive for downloading')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip running the data preprocessing script')
    args = parser.parse_args()
    output_dir = args.output_dir or "."
    output_path = Path(output_dir) / "recipes.json" 
    download_dataset(url=args.url, output_dir=args.output_dir, use_gdrive=not args.no_gdrive)
    
    if not args.skip_preprocessing:
        run_preprocessing()

    # ✅ Post-validation of the final recipes.json
    print("\n🔍 Verifying final recipes.json ...")
    try:
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise ValueError("recipes.json was not created or is empty.")

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list) or len(data) == 0:
                raise ValueError("recipes.json is not a valid non-empty list.")

        print(f"✅ recipes.json is valid with {len(data)} entries.")
    except Exception as e:
        print(f"❌ Final dataset validation failed: {e}")
        print("You may need to check data_preprocessing.py or the download source.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
