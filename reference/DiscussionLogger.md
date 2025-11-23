# Logger class for cell type annotation discussions

Logger class for cell type annotation discussions

Logger class for cell type annotation discussions

## Public fields

- `log_dir`:

  Directory for storing log files

- `current_log`:

  Current log file handle

- `session_id`:

  Unique identifier for the current session

## Methods

### Public methods

- [`DiscussionLogger$new()`](#method-DiscussionLogger-new)

- [`DiscussionLogger$start_cluster_discussion()`](#method-DiscussionLogger-start_cluster_discussion)

- [`DiscussionLogger$log_entry()`](#method-DiscussionLogger-log_entry)

- [`DiscussionLogger$log_prediction()`](#method-DiscussionLogger-log_prediction)

- [`DiscussionLogger$log_consensus_check()`](#method-DiscussionLogger-log_consensus_check)

- [`DiscussionLogger$log_final_consensus()`](#method-DiscussionLogger-log_final_consensus)

- [`DiscussionLogger$end_cluster_discussion()`](#method-DiscussionLogger-end_cluster_discussion)

- [`DiscussionLogger$clone()`](#method-DiscussionLogger-clone)

------------------------------------------------------------------------

### Method `new()`

Initialize a new logger

#### Usage

    DiscussionLogger$new(base_dir = "logs")

#### Arguments

- `base_dir`:

  Base directory for logs

------------------------------------------------------------------------

### Method `start_cluster_discussion()`

Start logging a new cluster discussion

#### Usage

    DiscussionLogger$start_cluster_discussion(
      cluster_id,
      tissue_name = NULL,
      marker_genes
    )

#### Arguments

- `cluster_id`:

  Cluster identifier

- `tissue_name`:

  Tissue name

- `marker_genes`:

  List of marker genes

------------------------------------------------------------------------

### Method `log_entry()`

Log a discussion entry

#### Usage

    DiscussionLogger$log_entry(event_type, content)

#### Arguments

- `event_type`:

  Type of event

- `content`:

  Content to log

------------------------------------------------------------------------

### Method `log_prediction()`

Log a model's prediction

#### Usage

    DiscussionLogger$log_prediction(model_name, round_number, prediction)

#### Arguments

- `model_name`:

  Name of the model

- `round_number`:

  Discussion round number

- `prediction`:

  Model's prediction and reasoning

------------------------------------------------------------------------

### Method `log_consensus_check()`

Log consensus check results

#### Usage

    DiscussionLogger$log_consensus_check(
      round_number,
      reached_consensus,
      consensus_proportion,
      entropy = NULL
    )

#### Arguments

- `round_number`:

  Round number

- `reached_consensus`:

  Whether consensus was reached

- `consensus_proportion`:

  Proportion of models supporting the majority prediction

- `entropy`:

  Shannon entropy of the predictions (optional)

------------------------------------------------------------------------

### Method `log_final_consensus()`

Log final consensus result

#### Usage

    DiscussionLogger$log_final_consensus(final_cell_type, summary)

#### Arguments

- `final_cell_type`:

  Final determined cell type

- `summary`:

  Summary of the discussion

------------------------------------------------------------------------

### Method `end_cluster_discussion()`

End current cluster discussion and close log file

#### Usage

    DiscussionLogger$end_cluster_discussion()

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    DiscussionLogger$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
