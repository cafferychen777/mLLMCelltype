% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/logger.R
\name{DiscussionLogger}
\alias{DiscussionLogger}
\title{Logger class for cell type annotation discussions}
\description{
Logger class for cell type annotation discussions

Logger class for cell type annotation discussions
}
\section{Public fields}{
\if{html}{\out{<div class="r6-fields">}}
\describe{
\item{\code{log_dir}}{Directory for storing log files}

\item{\code{current_log}}{Current log file handle}

\item{\code{session_id}}{Unique identifier for the current session}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-DiscussionLogger-new}{\code{DiscussionLogger$new()}}
\item \href{#method-DiscussionLogger-start_cluster_discussion}{\code{DiscussionLogger$start_cluster_discussion()}}
\item \href{#method-DiscussionLogger-log_entry}{\code{DiscussionLogger$log_entry()}}
\item \href{#method-DiscussionLogger-log_prediction}{\code{DiscussionLogger$log_prediction()}}
\item \href{#method-DiscussionLogger-log_consensus_check}{\code{DiscussionLogger$log_consensus_check()}}
\item \href{#method-DiscussionLogger-log_final_consensus}{\code{DiscussionLogger$log_final_consensus()}}
\item \href{#method-DiscussionLogger-end_cluster_discussion}{\code{DiscussionLogger$end_cluster_discussion()}}
\item \href{#method-DiscussionLogger-clone}{\code{DiscussionLogger$clone()}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-DiscussionLogger-new"></a>}}
\if{latex}{\out{\hypertarget{method-DiscussionLogger-new}{}}}
\subsection{Method \code{new()}}{
Initialize a new logger
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DiscussionLogger$new(base_dir = "logs")}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{base_dir}}{Base directory for logs}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-DiscussionLogger-start_cluster_discussion"></a>}}
\if{latex}{\out{\hypertarget{method-DiscussionLogger-start_cluster_discussion}{}}}
\subsection{Method \code{start_cluster_discussion()}}{
Start logging a new cluster discussion
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DiscussionLogger$start_cluster_discussion(
  cluster_id,
  tissue_name = NULL,
  marker_genes
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{cluster_id}}{Cluster identifier}

\item{\code{tissue_name}}{Tissue name}

\item{\code{marker_genes}}{List of marker genes}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-DiscussionLogger-log_entry"></a>}}
\if{latex}{\out{\hypertarget{method-DiscussionLogger-log_entry}{}}}
\subsection{Method \code{log_entry()}}{
Log a discussion entry
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DiscussionLogger$log_entry(event_type, content)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{event_type}}{Type of event}

\item{\code{content}}{Content to log}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-DiscussionLogger-log_prediction"></a>}}
\if{latex}{\out{\hypertarget{method-DiscussionLogger-log_prediction}{}}}
\subsection{Method \code{log_prediction()}}{
Log a model's prediction
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DiscussionLogger$log_prediction(model_name, round_number, prediction)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{model_name}}{Name of the model}

\item{\code{round_number}}{Discussion round number}

\item{\code{prediction}}{Model's prediction and reasoning}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-DiscussionLogger-log_consensus_check"></a>}}
\if{latex}{\out{\hypertarget{method-DiscussionLogger-log_consensus_check}{}}}
\subsection{Method \code{log_consensus_check()}}{
Log consensus check results
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DiscussionLogger$log_consensus_check(
  round_number,
  reached_consensus,
  consensus_proportion,
  entropy = NULL
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{round_number}}{Round number}

\item{\code{reached_consensus}}{Whether consensus was reached}

\item{\code{consensus_proportion}}{Proportion of models supporting the majority prediction}

\item{\code{entropy}}{Shannon entropy of the predictions (optional)}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-DiscussionLogger-log_final_consensus"></a>}}
\if{latex}{\out{\hypertarget{method-DiscussionLogger-log_final_consensus}{}}}
\subsection{Method \code{log_final_consensus()}}{
Log final consensus result
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DiscussionLogger$log_final_consensus(final_cell_type, summary)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{final_cell_type}}{Final determined cell type}

\item{\code{summary}}{Summary of the discussion}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-DiscussionLogger-end_cluster_discussion"></a>}}
\if{latex}{\out{\hypertarget{method-DiscussionLogger-end_cluster_discussion}{}}}
\subsection{Method \code{end_cluster_discussion()}}{
End current cluster discussion and close log file
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DiscussionLogger$end_cluster_discussion()}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-DiscussionLogger-clone"></a>}}
\if{latex}{\out{\hypertarget{method-DiscussionLogger-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DiscussionLogger$clone(deep = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{deep}}{Whether to make a deep clone.}
}
\if{html}{\out{</div>}}
}
}
}
