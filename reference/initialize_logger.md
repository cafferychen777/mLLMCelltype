# Reinitialize global logger with a specific directory

Preserves the current logger configuration (level, size, retention,
console/json) while changing the log directory for a new annotation
session.

## Usage

``` r
initialize_logger(log_dir = "logs")
```

## Arguments

- log_dir:

  Directory for log files

## Value

Invisible logger object
