import pandas as pd
import os

def process_tissue_markers(df, tissue, output_dir):
    """Process markers for a specific tissue and save to CSV."""
    # Create output path for this tissue
    output_path = os.path.join(output_dir, f"GTEx_{tissue.lower().replace(' ', '_')}_markers.csv")

    # Filter for current tissue
    tissue_df = df[df['tissue'] == tissue].copy()

    # Create a list to store results
    results = []

    # Process each cell type
    for cell_type in sorted(tissue_df['celltype'].unique()):
        # Get data for current cell type
        cell_type_data = tissue_df[tissue_df['celltype'] == cell_type].copy()

        # Sort by absolute log2FC value to get top markers
        cell_type_data['abs_log2FC'] = abs(cell_type_data['log2FC'])
        top_markers = cell_type_data.nlargest(10, 'abs_log2FC')

        # Get unique genes (up to 10)
        unique_genes = list(top_markers['gene'].unique())[:10]

        # Add to results
        results.append({
            'cell_type': cell_type,
            'gene': ','.join(unique_genes)
        })

    # Write to CSV file directly
    with open(output_path, 'w') as f:
        f.write('cell_type,gene\n')
        for row in results:
            f.write(f"{row['cell_type']},{row['gene']}\n")

    print(f"\nProcessed {tissue}:")
    print(f"Saved to: {output_path}")
    print("\nCell types and their markers:")

    # Display results
    for row in results:
        print(f"\n{row['cell_type']}:")
        print(row['gene'].replace(',', ', '))

def main():
    # Set paths
    excel_path = "data/reference/science.abl4290_table_s2.xlsx"
    output_dir = "data/reference"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the Excel file, skipping the first row which contains the long title
    df = pd.read_excel(excel_path, skiprows=1)

    # Process each tissue
    for tissue in sorted(df['tissue'].unique()):
        process_tissue_markers(df, tissue, output_dir)

if __name__ == "__main__":
    main()
