#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import os
import glob

# List of already processed datasets
PROCESSED_DATASETS = {
    'Thymus',
    'Tongue',
    'Trachea',
    'Uterus',
    'Vasculature'
}

def process_dataset(h5ad_path):
    """
    Process a single h5ad dataset to find marker genes.

    Args:
        h5ad_path (str): Path to the h5ad file
    """
    # Get tissue name from file path
    tissue = os.path.basename(h5ad_path).replace('_filtered.h5ad', '').replace('TS_', '')

    # Skip if already processed
    if tissue in PROCESSED_DATASETS:
        print(f"\nSkipping {tissue} dataset (already processed)")
        return

    print(f"\n{'='*50}")
    print(f"Processing {tissue} dataset...")
    print(f"{'='*50}")

    try:
        # Load the data
        print("Loading dataset...")
        adata = sc.read_h5ad(h5ad_path)

        # Create a mapping from ensemblid to feature_name
        print("\nCreating gene mapping...")
        # Remove version numbers from Ensembl IDs (e.g., ENSG00000223972.5 -> ENSG00000223972)
        gene_map = pd.Series(adata.var['feature_name'].values,
                            index=[eid.split('.')[0] for eid in adata.var['ensemblid'].values]).to_dict()

        # Normalize the data
        print("\nNormalizing data...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Calculate marker genes for each cell type
        print("Calculating marker genes...")
        sc.tl.rank_genes_groups(adata, 'cell_type', method='wilcoxon')

        # Get cell types
        cell_types = adata.obs['cell_type'].unique()

        # Create a list to store results
        marker_lines = []

        # Extract top 10 markers for each cell type
        print("\nExtracting top markers...")
        for cell_type in cell_types:
            markers = sc.get.rank_genes_groups_df(adata, group=cell_type)
            # Convert Ensembl IDs to gene symbols
            top_10_markers = []
            for gene_id in markers['names'].head(10):
                # Remove version number if present
                base_id = gene_id.split('.')[0]
                symbol = gene_map.get(base_id, gene_id)
                top_10_markers.append(symbol)

            # Create the line in the required format
            marker_line = ','.join([cell_type] + top_10_markers)
            marker_lines.append(marker_line)
            print(f"{cell_type}: {', '.join(top_10_markers)}")

        # Save to CSV
        output_dir = 'data/reference'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'TS_{tissue}_markers.csv')

        print(f"\nSaving results to {output_file}")
        with open(output_file, 'w') as f:
            f.write('\n'.join(marker_lines))

        print("\nDone! The markers have been saved in the format:")
        print("celltype,gene1,gene2,gene3,...")

    except Exception as e:
        print(f"\nError processing {tissue} dataset:")
        print(str(e))

def main():
    """
    Process all datasets in the raw directory.
    """
    # Get all h5ad files
    data_dir = 'data/raw'
    h5ad_files = glob.glob(os.path.join(data_dir, 'TS_*_filtered.h5ad'))

    # Sort files to process them in a consistent order
    h5ad_files.sort()

    # Filter out already processed datasets
    remaining_files = [f for f in h5ad_files if os.path.basename(f).replace('_filtered.h5ad', '').replace('TS_', '') not in PROCESSED_DATASETS]

    print(f"Found {len(remaining_files)} datasets to process:")
    for f in remaining_files:
        tissue = os.path.basename(f).replace('_filtered.h5ad', '').replace('TS_', '')
        print(f"- {tissue}")

    # Process each dataset
    for h5ad_file in remaining_files:
        process_dataset(h5ad_file)

if __name__ == "__main__":
    main()
