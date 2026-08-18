#!/usr/bin/env python3
"""
Download Tabula Sapiens v2 dataset from Figshare.

Tabula Sapiens is a multiple-organ, single-cell transcriptomic atlas of humans.
Figshare Article: https://figshare.com/articles/dataset/27921984

Note: The complete dataset is quite large (~53 GB).

Usage:
    python download_tabula_sapiens.py [--output-dir /path/to/output]
"""

import argparse
import zipfile
from pathlib import Path

import requests


def download_file(url, output_file, chunk_size=8192):
    """Download a file from URL to the specified output file with progress."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    written = 0

    print(f"Downloading to {output_file}")
    with open(output_file, "wb") as f:
        for data in response.iter_content(chunk_size):
            written += len(data)
            f.write(data)
            if total_size > 0:
                progress = int((written / total_size) * 100)
                print(f"\rProgress: {progress}%", end="")
    print()  # New line after progress


def download_tabula_sapiens(data_dir):
    """
    Download Tabula Sapiens v2 dataset from Figshare.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Figshare direct download URL for the complete dataset
    download_url = "https://figshare.com/ndownloader/articles/27921984/versions/1"
    zip_file = data_dir / "tabula_sapiens_v2.zip"

    try:
        print("Downloading Tabula Sapiens v2 dataset (~53 GB)...")
        print("Note: This is a large download and may take a while.")
        download_file(download_url, zip_file)

        print("\nExtracting files...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        # Remove the zip file after extraction
        zip_file.unlink()

        print(f"\nFiles have been downloaded and extracted to: {data_dir.absolute()}")
        print("The dataset includes:")
        print("- Complete dataset in h5ad format")
        print("- Cell metadata")
        print("- Additional supplementary files")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading files: {e!s}")
        raise
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file")
        if zip_file.exists():
            zip_file.unlink()
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e!s}")
        if zip_file.exists():
            zip_file.unlink()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Tabula Sapiens v2 dataset from Figshare"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/tabula_sapiens",
        help="Output directory for downloaded files (default: ./data/tabula_sapiens)",
    )
    args = parser.parse_args()

    print(f"Output directory: {args.output_dir}")
    download_tabula_sapiens(args.output_dir)
