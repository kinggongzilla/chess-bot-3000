#!/usr/bin/env python3
"""
Download Lichess Chess Games Dataset (skip existing, resume only if empty)
Downloads the parquet files maintaining the original structure
"""
from huggingface_hub import HfFileSystem
import os
import requests
from pathlib import Path

# Configuration
OUTPUT_DIR = "./lichess_data"
YEAR_START = 2023
YEAR_END = 2024
REPO_ID = "Lichess/standard-chess-games"

print("=" * 60)
print("Lichess Standard Chess Games Dataset Downloader")
print("=" * 60)
print(f"\nYears: {YEAR_START}-{YEAR_END} (inclusive)")
print(f"Output directory: {OUTPUT_DIR}/")
print("\nThis will download parquet files maintaining the structure:")
print("  data/year=YYYY/month=MM/train-*.parquet")
print("=" * 60)
print()

# Initialize HuggingFace filesystem
fs = HfFileSystem()

# Find all parquet files for the specified years
print("Finding files to download...\n")
files_to_download = []
for year in range(YEAR_START, YEAR_END + 1):
    for month in range(1, 13):
        pattern = f"datasets/{REPO_ID}/data/year={year}/month={month:02d}"
        try:
            files = fs.ls(pattern, detail=False)
            parquet_files = [f for f in files if f.endswith('.parquet')]
            files_to_download.extend(parquet_files)
            if parquet_files:
                print(f"  Found {len(parquet_files)} files for {year}-{month:02d}")
        except:
            # Month doesn't exist, skip
            pass

print(f"\nTotal files to download: {len(files_to_download)}")
if not files_to_download:
    print("No files found for the specified years!")
    exit(1)

# Download files
print("\nDownloading files...\n")
for i, file_path in enumerate(files_to_download, 1):
    rel_path = file_path.replace(f"datasets/{REPO_ID}/", "")
    local_path = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Skip if file exists and is not empty
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"[{i}/{len(files_to_download)}] {rel_path} already exists, skipping...")
        continue

    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{rel_path}"
    print(f"[{i}/{len(files_to_download)}] Downloading {rel_path}...")

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        file_size = os.path.getsize(local_path) / (1024**2)  # MB
        print(f"  ✓ Saved ({file_size:.1f} MB)")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print(f"\n✓ Download complete!")
print(f"✓ Files stored in: {OUTPUT_DIR}/")
print("\nDirectory structure:")
print(f"  {OUTPUT_DIR}/data/year=YYYY/month=MM/train-*.parquet")
