from wildlife_datasets import datasets
import os

# Ensure the directory exists
os.makedirs('data', exist_ok=True)

# Download and set up the dataset
print("Downloading dataset...")
datasets.MacaqueFaces.get_data('data/MacaqueFaces')
print("Download complete!")