% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/prompt_templates.R
\name{create_annotation_prompt}
\alias{create_annotation_prompt}
\title{Prompt templates for mLLMCelltype}
\usage{
create_annotation_prompt(input, tissue_name, top_gene_count = 10)
}
\arguments{
\item{input}{Either the differential gene table returned by Seurat FindAllMarkers() function, or a list of genes}

\item{tissue_name}{The name of the tissue}

\item{top_gene_count}{Number of top differential genes to use per cluster}
}
\value{
A list containing the prompt string and expected count of responses
}
\description{
This file contains all prompt template functions used in mLLMCelltype.
These functions create various prompts for different stages of the cell type annotation process.
Create prompt for cell type annotation
}
