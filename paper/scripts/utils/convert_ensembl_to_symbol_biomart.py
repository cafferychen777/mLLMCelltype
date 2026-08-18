import pandas as pd
from pybiomart import Server

def get_ensembl_mapping():
    # Connect to Ensembl BioMart
    server = Server(host='http://www.ensembl.org')
    dataset = (server.marts['ENSEMBL_MART_ENSEMBL']
                    .datasets['hsapiens_gene_ensembl'])

    # Get all ENSEMBL IDs and their corresponding gene symbols
    data = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])
    # Convert to dictionary for faster lookup
    return dict(zip(data['Gene stable ID'], data['Gene name']))

print("Reading the markers file...")
# Read the CSV file
df = pd.read_csv('data/reference/M1C_markers.csv', header=None)

print("Getting ENSEMBL to gene symbol mapping...")
ensembl_to_symbol = get_ensembl_mapping()

print("Converting ENSEMBL IDs to gene symbols...")
# Convert each ENSEMBL ID to gene symbol
for col in range(1, 11):  # Skip the first column as it contains cluster names
    print(f"Processing column {col}/10...")
    df[col] = df[col].apply(lambda x: ensembl_to_symbol.get(x, x))

print("Saving the converted file...")
# Save the converted file
output_file = 'data/reference/M1C_markers_symbols.csv'
df.to_csv(output_file, header=False, index=False)
print(f"Converted file saved to: {output_file}")

# Display first few rows
print("\nFirst few rows of the converted file:")
print(df.head())
