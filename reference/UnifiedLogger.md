# Unified Logger for mLLMCelltype Package

Unified Logger for mLLMCelltype Package

Unified Logger for mLLMCelltype Package

## Details

This logger provides centralized, multi-level logging with structured
output, log rotation, and performance monitoring capabilities.

## Public fields

- `log_dir`:

  Directory for storing log files

- `log_level`:

  Current logging level

- `session_id`:

  Unique identifier for the current session

- `max_log_size`:

  Maximum log file size in MB (default: 10MB)

- `max_log_files`:

  Maximum number of log files to keep (default: 5)

- `enable_console`:

  Whether to output to console (default: FALSE)

- `enable_json`:

  Whether to use JSON format (default: TRUE)

- `performance_stats`:

  Performance monitoring statistics

## Methods

### Public methods

- [`UnifiedLogger$new()`](#method-UnifiedLogger-new)

- [`UnifiedLogger$debug()`](#method-UnifiedLogger-debug)

- [`UnifiedLogger$info()`](#method-UnifiedLogger-info)

- [`UnifiedLogger$warn()`](#method-UnifiedLogger-warn)

- [`UnifiedLogger$error()`](#method-UnifiedLogger-error)

- [`UnifiedLogger$log_api_call()`](#method-UnifiedLogger-log_api_call)

- [`UnifiedLogger$log_api_request_response()`](#method-UnifiedLogger-log_api_request_response)

- [`UnifiedLogger$log_cache_operation()`](#method-UnifiedLogger-log_cache_operation)

- [`UnifiedLogger$log_cluster_progress()`](#method-UnifiedLogger-log_cluster_progress)

- [`UnifiedLogger$log_discussion()`](#method-UnifiedLogger-log_discussion)

- [`UnifiedLogger$log_model_response()`](#method-UnifiedLogger-log_model_response)

- [`UnifiedLogger$get_performance_summary()`](#method-UnifiedLogger-get_performance_summary)

- [`UnifiedLogger$cleanup_logs()`](#method-UnifiedLogger-cleanup_logs)

- [`UnifiedLogger$set_level()`](#method-UnifiedLogger-set_level)

- [`UnifiedLogger$clone()`](#method-UnifiedLogger-clone)

------------------------------------------------------------------------

### Method `new()`

Initialize the unified logger

#### Usage

    UnifiedLogger$new(
      base_dir = "logs",
      level = "INFO",
      max_size = 10,
      max_files = 5,
      console_output = FALSE,
      json_format = TRUE
    )

#### Arguments

- `base_dir`:

  Directory for log files

- `level`:

  Minimum log level

- `max_size`:

  Maximum main log size in megabytes

- `max_files`:

  Maximum number of retained main logs

- `console_output`:

  Whether to mirror logs to the console

- `json_format`:

  Whether main logs use JSON format

------------------------------------------------------------------------

### Method [`debug()`](https://rdrr.io/r/base/debug.html)

Log a debug message

#### Usage

    UnifiedLogger$debug(message, context = NULL)

#### Arguments

- `message`:

  Log message

- `context`:

  Optional structured context

------------------------------------------------------------------------

### Method `info()`

Log an info message

#### Usage

    UnifiedLogger$info(message, context = NULL)

#### Arguments

- `message`:

  Log message

- `context`:

  Optional structured context

------------------------------------------------------------------------

### Method `warn()`

Log a warning message

#### Usage

    UnifiedLogger$warn(message, context = NULL)

#### Arguments

- `message`:

  Log message

- `context`:

  Optional structured context

------------------------------------------------------------------------

### Method `error()`

Log an error message

#### Usage

    UnifiedLogger$error(message, context = NULL)

#### Arguments

- `message`:

  Log message

- `context`:

  Optional structured context

------------------------------------------------------------------------

### Method `log_api_call()`

Log API call performance

#### Usage

    UnifiedLogger$log_api_call(
      provider,
      model,
      duration,
      success = TRUE,
      tokens = NULL
    )

#### Arguments

- `provider`:

  Provider identifier

- `model`:

  Model identifier

- `duration`:

  Request duration in seconds

- `success`:

  Whether the request succeeded

- `tokens`:

  Optional token usage metadata

------------------------------------------------------------------------

### Method `log_api_request_response()`

Log complete API request and response for debugging and audit

#### Usage

    UnifiedLogger$log_api_request_response(
      provider,
      model,
      prompt_content,
      response_content,
      request_metadata = NULL,
      response_metadata = NULL
    )

#### Arguments

- `provider`:

  Provider identifier

- `model`:

  Model identifier

- `prompt_content`:

  Request prompt

- `response_content`:

  Provider response or error text

- `request_metadata`:

  Optional request metadata

- `response_metadata`:

  Optional response metadata

------------------------------------------------------------------------

### Method `log_cache_operation()`

Log cache operations

#### Usage

    UnifiedLogger$log_cache_operation(operation, key, size = NULL)

#### Arguments

- `operation`:

  Cache operation name

- `key`:

  Cache key

- `size`:

  Optional cache object size in bytes

------------------------------------------------------------------------

### Method `log_cluster_progress()`

Log cluster annotation progress

#### Usage

    UnifiedLogger$log_cluster_progress(cluster_id, stage, progress = NULL)

#### Arguments

- `cluster_id`:

  Cluster identifier

- `stage`:

  Processing stage

- `progress`:

  Optional progress value

------------------------------------------------------------------------

### Method `log_discussion()`

Log detailed cluster discussion with complete model conversations

#### Usage

    UnifiedLogger$log_discussion(cluster_id, event_type, data = NULL)

#### Arguments

- `cluster_id`:

  Cluster identifier

- `event_type`:

  Discussion event type

- `data`:

  Optional event payload

------------------------------------------------------------------------

### Method `log_model_response()`

Log model response with concise summary in main log and full text in
file

#### Usage

    UnifiedLogger$log_model_response(
      provider,
      model,
      response,
      stage = "annotation",
      cluster_id = NULL
    )

#### Arguments

- `provider`:

  Provider identifier

- `model`:

  Model identifier

- `response`:

  Model response

- `stage`:

  Processing stage

- `cluster_id`:

  Optional cluster identifier

------------------------------------------------------------------------

### Method `get_performance_summary()`

Get performance summary

#### Usage

    UnifiedLogger$get_performance_summary()

------------------------------------------------------------------------

### Method `cleanup_logs()`

Clean up old log files

#### Usage

    UnifiedLogger$cleanup_logs(force = FALSE)

#### Arguments

- `force`:

  Whether to remove every main log file

------------------------------------------------------------------------

### Method `set_level()`

Set logging level

#### Usage

    UnifiedLogger$set_level(level)

#### Arguments

- `level`:

  Minimum log level

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    UnifiedLogger$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
