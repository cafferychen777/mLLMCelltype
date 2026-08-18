#!/usr/bin/env python3
"""
Download popV reference data from Zenodo.

This script downloads the popV pretrained model reference data from Zenodo.
Zenodo Record: https://zenodo.org/record/7587774

Usage:
    python download_zenodo_data.py [--output-dir /path/to/output]
"""

import argparse
import os

import requests


def download_zenodo_files(data_dir):
    """Download all files from the Zenodo record."""
    os.makedirs(data_dir, exist_ok=True)

    # Get the Zenodo record metadata
    res = requests.get("https://zenodo.org/api/records/7587774")
    if res.status_code != 200:
        print(f"Failed to get metadata: {res.status_code}")
        return

    # Extract file information
    files_data = res.json().get("files", [])

    print(f"Found {len(files_data)} files")

    # Download each file
    for file_info in files_data:
        filename = file_info["key"]
        download_url = file_info["links"]["self"]

        output_path = os.path.join(data_dir, filename)

        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"File {filename} already exists, skipping...")
            continue

        print(f"Downloading {filename}...")
        response = requests.get(download_url, stream=True)

        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.writelines(response.iter_content(chunk_size=8192))
            print(f"Successfully downloaded {filename}")
        else:
            print(f"Failed to download {filename}: {response.status_code}")


def check_data(data_dir):
    """Check and display information about downloaded files."""
    print("\nChecking downloaded data:")

    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist")
        return

    files = os.listdir(data_dir)
    print(f"Total files in directory: {len(files)}")

    # Print information about each .h5ad file
    for file in files:
        if file.endswith(".h5ad"):
            file_path = os.path.join(data_dir, file)
            print(f"\nAnalyzing {file}:")
            try:
                import scanpy as sc

                adata = sc.read_h5ad(file_path)
                print(f"  Shape: {adata.shape}")
                print(f"  Observations (cells): {adata.n_obs}")
                print(f"  Variables (genes): {adata.n_vars}")
                print(f"  Available annotations: {list(adata.obs.columns)[:5]}...")
            except ImportError:
                print("  (Install scanpy to view file details)")
            except Exception as e:  # noqa: BLE001 - report unreadable optional files
                print(f"  Error reading {file}: {e!s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download popV reference data from Zenodo"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/popv_data",
        help="Output directory for downloaded files (default: ./data/popv_data)",
    )
    args = parser.parse_args()

    print("Starting download process...")
    print(f"Output directory: {args.output_dir}")
    download_zenodo_files(args.output_dir)
    print("\nDownload complete. Starting data check...")
    check_data(args.output_dir)
