# Create reasoning-aware prompt for cell type annotation

Create reasoning-aware prompt for cell type annotation

## Usage

``` r
create_reasoning_annotation_prompt(input, tissue_name, top_gene_count = 10)
```

## Arguments

- input:

  Either a data frame from Seurat's FindAllMarkers() or a list for each
  cluster where each element is either a character vector of genes or a
  list containing a `genes` field.

- tissue_name:

  Tissue context for the annotation (e.g., 'human PBMC', 'mouse brain')

- top_gene_count:

  Number of top genes to use per cluster when input is from Seurat.
  Default: 10

## Value

A list with `prompt` (formatted prompt text), `expected_count` (number
of clusters), and `gene_lists` (cluster ID to marker genes mapping).
