% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/consensus_annotation.R
\name{get_initial_predictions}
\alias{get_initial_predictions}
\title{Get initial predictions from all models}
\usage{
get_initial_predictions(input, tissue_name, models, api_keys, top_gene_count)
}
\arguments{
\item{input}{Either the differential gene table or a list of genes}

\item{tissue_name}{The tissue type or cell source}

\item{models}{Vector of model names to use}

\item{api_keys}{Named list of API keys}

\item{top_gene_count}{Number of top differential genes to use}
}
\value{
A list containing individual predictions and successful models
}
\description{
This function retrieves initial cell type predictions from all specified models.
It is an internal helper function used by the interactive_consensus_annotation function.
}
\keyword{internal}
