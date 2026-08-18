#!/usr/bin/env python3
"""
Convert Ensembl IDs to Gene Symbols

This script converts Ensembl gene IDs to gene symbols using the mygene package.
Useful for processing marker gene files that use Ensembl IDs.

Reference: mygene - https://pypi.org/project/mygene/

Usage:
    python convert_ensembl_to_symbol.py --input markers.csv --output markers_symbols.csv
"""

import argparse

import pandas as pd


def convert_ensembl_to_symbol(ensembl_id, mg=None):
    """
    Convert a single Ensembl ID to gene symbol.

    Parameters
    ----------
    ensembl_id : str
        Ensembl gene ID (e.g., "ENSG00000141510")
    mg : mygene.MyGeneInfo, optional
        MyGeneInfo instance for reuse

    Returns
    -------
    str
        Gene symbol or original ID if conversion fails
    """
    try:
        import mygene

        if mg is None:
            mg = mygene.MyGeneInfo()
        out = mg.getgene(ensembl_id, fields="symbol")
        return out.get("symbol", ensembl_id) if out else ensembl_id
    except Exception:  # noqa: BLE001 - preserve the input when lookup fails
        return ensembl_id


def batch_convert_ensembl_to_symbol(ensembl_ids):
    """
    Convert a list of Ensembl IDs to gene symbols in batch.

    Parameters
    ----------
    ensembl_ids : list
        List of Ensembl gene IDs

    Returns
    -------
    dict
        Mapping from Ensembl ID to gene symbol
    """
    try:
        import mygene

        mg = mygene.MyGeneInfo()

        # Remove duplicates and NaN
        unique_ids = list(
            {x for x in ensembl_ids if pd.notna(x) and str(x).startswith("ENSG")}
        )

        if not unique_ids:
            return {}

        print(f"Querying {len(unique_ids)} unique Ensembl IDs...")
        results = mg.querymany(
            unique_ids, scopes="ensembl.gene", fields="symbol", species="human"
        )

        # Build mapping
        mapping = {}
        for r in results:
            query = r.get("query", "")
            symbol = r.get("symbol", query)
            mapping[query] = symbol if symbol else query

        return mapping

    except ImportError:
        print("Warning: mygene not installed. Install with: pip install mygene")
        return {}
    except Exception as e:  # noqa: BLE001 - batch lookup failures are non-fatal
        print(f"Error in batch conversion: {e}")
        return {}


def convert_marker_file(input_file, output_file, gene_columns=None, batch_mode=True):
    """
    Convert Ensembl IDs in a marker gene CSV file to gene symbols.

    Parameters
    ----------
    input_file : str
        Path to input CSV file
    output_file : str
        Path to save converted file
    gene_columns : list, optional
        Column indices containing gene names (0-indexed).
        If None, attempts to detect automatically.
    batch_mode : bool
        Use batch conversion for speed (default: True)
    """
    print(f"Reading: {input_file}")
    df = pd.read_csv(input_file, header=None)
    print(f"Shape: {df.shape}")

    # Detect gene columns if not specified
    if gene_columns is None:
        # Assume first column is cluster names, rest are genes
        gene_columns = list(range(1, df.shape[1]))
        print(f"Auto-detected gene columns: {gene_columns}")

    if batch_mode:
        # Collect all Ensembl IDs
        all_ensembl_ids = []
        for col in gene_columns:
            all_ensembl_ids.extend(df[col].tolist())

        # Batch convert
        mapping = batch_convert_ensembl_to_symbol(all_ensembl_ids)

        # Apply mapping
        for col in gene_columns:
            df[col] = df[col].apply(lambda x: mapping.get(x, x) if pd.notna(x) else x)
    else:
        # Convert one by one (slower but more reliable)
        try:
            import mygene

            mg = mygene.MyGeneInfo()
        except ImportError:
            print("Error: mygene not installed. Install with: pip install mygene")
            return

        for col in gene_columns:
            print(f"Processing column {col}...")
            df[col] = df[col].apply(lambda x: convert_ensembl_to_symbol(x, mg))

    # Save result
    print(f"Saving to: {output_file}")
    df.to_csv(output_file, header=False, index=False)

    # Display sample
    print("\nFirst few rows of converted file:")
    print(df.head())


def main():
    parser = argparse.ArgumentParser(
        description="Convert Ensembl IDs to gene symbols in marker files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert marker file
    python convert_ensembl_to_symbol.py --input markers.csv --output markers_symbols.csv

    # Specify gene columns (0-indexed)
    python convert_ensembl_to_symbol.py --input data.csv --output out.csv --columns 1 2 3 4 5

Requirements:
    pip install mygene pandas
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input CSV file with Ensembl IDs"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to save converted CSV file"
    )
    parser.add_argument(
        "--columns",
        type=int,
        nargs="+",
        default=None,
        help="Column indices containing genes (0-indexed). If not specified, auto-detects.",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch mode (slower but more reliable)",
    )

    args = parser.parse_args()

    convert_marker_file(
        input_file=args.input,
        output_file=args.output,
        gene_columns=args.columns,
        batch_mode=not args.no_batch,
    )


if __name__ == "__main__":
    main()
