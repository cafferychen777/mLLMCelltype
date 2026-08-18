import pandas as pd
import os
from pathlib import Path

# Directory containing the supplementary tables
supp_dir = Path("data/reference/NIHMS1828607-supplement-Supplementary_Tables")

def read_excel_info(file_path):
    try:
        # Read Excel file
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        print(f"\nFile: {file_path.name}")
        print("Sheet names:", sheet_names)

        # Read each sheet and print first few rows and columns
        for sheet in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)
            print(f"\nSheet: {sheet}")
            print("Columns:", df.columns.tolist())
            print("\nFirst few rows:")
            print(df.head(3))
            print("-" * 80)

    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")

# Process each Excel file
for excel_file in supp_dir.glob("*.xlsx"):
    read_excel_info(excel_file)
