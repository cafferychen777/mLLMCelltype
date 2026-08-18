import pandas as pd
import scanpy as sc
import numpy as np
import os
import gc
from scipy import sparse

# Set paths
base_dir = 'data/processed'
annotation_file = os.path.join(base_dir, 'GSE131907_Lung_Cancer_cell_annotation.txt')
expression_file = os.path.join(base_dir, 'GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz')
output_file = os.path.join(base_dir, 'GSE131907_Lung_Cancer.h5ad')

# Read cell annotations
print("Reading cell annotations...")
cell_annotations = pd.read_csv(annotation_file, sep='\t', index_col='Barcode')

# Read expression matrix in chunks
print("Reading expression matrix in chunks...")
chunk_size = 1000  # Adjust this based on available memory
reader = pd.read_csv(expression_file, sep='\t', compression='gzip', chunksize=chunk_size, index_col=0)

# Get the first chunk to initialize the matrix
first_chunk = next(reader)
n_genes = len(first_chunk.index)
n_cells = len(first_chunk.columns)

# Create a sparse matrix to store the data
print(f"Initializing sparse matrix with shape: ({n_cells}, {n_genes})")
expression_matrix = sparse.lil_matrix((n_cells, n_genes), dtype=np.float32)

# Process the first chunk
gene_names = first_chunk.index.tolist()
expression_matrix[:, :len(first_chunk.index)] = first_chunk.values.T

# Process remaining chunks
current_position = len(first_chunk.index)
for chunk in reader:
    print(f"Processing chunk at position {current_position}...")
    chunk_size = len(chunk.index)
    end_position = current_position + chunk_size
    expression_matrix[:, current_position:end_position] = chunk.values.T
    current_position = end_position
    gc.collect()

# Convert to CSR format for better memory efficiency
print("Converting to CSR format...")
expression_matrix = expression_matrix.tocsr()

# Create AnnData object
print("Creating AnnData object...")
adata = sc.AnnData(X=expression_matrix,
                  obs=cell_annotations,
                  var=pd.DataFrame(index=gene_names))

# Add some basic metadata
adata.uns['dataset'] = 'GSE131907_Lung_Cancer'
adata.uns['data_type'] = 'normalized_log2TPM'

# Save the h5ad file
print("Saving h5ad file...")
adata.write(output_file)
print(f"Saved h5ad file to: {output_file}")
