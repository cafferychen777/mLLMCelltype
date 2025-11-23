# Prompt templates for mLLMCelltype

This file contains all prompt template functions used in mLLMCelltype.
These functions create various prompts for different stages of the cell
type annotation process. Create prompt for cell type annotation

## Usage

``` r
create_annotation_prompt(input, tissue_name, top_gene_count = 10)
```

## Arguments

- input:

  Either a data frame from Seurat's FindAllMarkers() or a list with
  'genes' field for each cluster

- tissue_name:

  Tissue context for the annotation (e.g., 'human PBMC', 'mouse brain')

- top_gene_count:

  Number of top genes to use per cluster when input is from Seurat.
  Default: 10

## Value

Character string containing the formatted prompt
