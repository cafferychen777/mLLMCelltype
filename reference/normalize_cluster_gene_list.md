# Prompt templates for mLLMCelltype

This file contains all prompt template functions used in mLLMCelltype.
These functions create various prompts for different stages of the cell
type annotation process. Normalize list input into a canonical
cluster-\>genes mapping

## Usage

``` r
normalize_cluster_gene_list(input)
```

## Arguments

- input:

  List input for cluster annotation

## Value

Named list of character vectors (cluster_id -\> genes)

## Details

For list input, each element can be either:

1.  a list containing a `genes` field, or

2.  a character vector of genes.

Naming rules:

- unnamed lists are assigned 0-based IDs ("0", "1", ...)

- fully numeric names are canonicalized; if minimum index is \>= 1, they
  are shifted to 0-based

- non-numeric names are preserved as-is
