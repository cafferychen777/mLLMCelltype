# Extract one line's annotation content for positional mapping

Returns `NULL` only for lines that do not occupy a cluster slot: an
empty-value preamble/header (`"Notes:"`) and an explicit label for an
unrequested/nonexistent cluster (`"Cluster 999: Noise"`). Every other
line – including a sentinel such as `"Unknown"` – keeps its slot so the
line-\>cluster correspondence is preserved (the caller leaves a non-real
slot unassigned). A numbered-list ordinal (`"1. T cells"`) or a
resolvable label prefix (`"0: T cells"`) is stripped to its content; a
colon that is part of the annotation (`"Neurons: excitatory"`) is kept
whole.

## Usage

``` r
extract_positional_annotation(line, all_clusters)
```

## Arguments

- line:

  Single response line.

- all_clusters:

  Character vector of requested cluster IDs.

## Value

Annotation content string, or `NULL` to drop the line.
