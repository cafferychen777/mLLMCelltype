# Cache Manager Class

Manages caching of consensus analysis results

## Public fields

- `cache_dir`:

  Directory to store cache files. Options:

  - NULL (default): Uses system cache directory

  - "local": Uses .mllmcelltype_cache in current directory

  - "temp": Uses temporary directory

  - Custom path: Any other string is used as directory path

- `cache_version`:

  Current cache version

## Methods

### Public methods

- [`CacheManager$new()`](#method-CacheManager-new)

- [`CacheManager$get_cache_dir()`](#method-CacheManager-get_cache_dir)

- [`CacheManager$generate_key()`](#method-CacheManager-generate_key)

- [`CacheManager$save_to_cache()`](#method-CacheManager-save_to_cache)

- [`CacheManager$load_from_cache()`](#method-CacheManager-load_from_cache)

- [`CacheManager$has_cache()`](#method-CacheManager-has_cache)

- [`CacheManager$get_cache_stats()`](#method-CacheManager-get_cache_stats)

- [`CacheManager$clear_cache()`](#method-CacheManager-clear_cache)

- [`CacheManager$validate_cache()`](#method-CacheManager-validate_cache)

- [`CacheManager$clone()`](#method-CacheManager-clone)

------------------------------------------------------------------------

### Method `new()`

Initialize cache manager

- NULL (default): Uses system cache directory via
  [`tools::R_user_dir()`](https://rdrr.io/r/tools/userdir.html)

- "local": Uses .mllmcelltype_cache in current directory

- "temp": Uses temporary directory (cleared on R restart)

- Custom path: Any other string is used as directory path

#### Usage

    CacheManager$new(cache_dir = NULL)

#### Arguments

- `cache_dir`:

  Cache directory selector or custom path

------------------------------------------------------------------------

### Method `get_cache_dir()`

Get actual cache directory path

#### Usage

    CacheManager$get_cache_dir()

------------------------------------------------------------------------

### Method `generate_key()`

Generate cache key from input parameters (improved version)

#### Usage

    CacheManager$generate_key(
      input,
      models,
      cluster_id,
      tissue_name = "",
      top_gene_count = 10,
      discussion_context = NULL
    )

#### Arguments

- `input`:

  Marker gene input

- `models`:

  Model identifiers contributing to the result

- `cluster_id`:

  Cluster identifier

- `tissue_name`:

  Tissue context included in the cache key

- `top_gene_count`:

  Number of marker genes used

- `discussion_context`:

  Optional normalized inputs that affect a discussion

------------------------------------------------------------------------

### Method `save_to_cache()`

Save results to cache

#### Usage

    CacheManager$save_to_cache(key, data)

#### Arguments

- `key`:

  Valid cache key

- `data`:

  Result object to persist

------------------------------------------------------------------------

### Method `load_from_cache()`

Load results from cache

#### Usage

    CacheManager$load_from_cache(key)

#### Arguments

- `key`:

  Valid cache key

------------------------------------------------------------------------

### Method `has_cache()`

Check if results exist in cache

#### Usage

    CacheManager$has_cache(key)

#### Arguments

- `key`:

  Valid cache key

------------------------------------------------------------------------

### Method `get_cache_stats()`

Get cache statistics

#### Usage

    CacheManager$get_cache_stats()

------------------------------------------------------------------------

### Method `clear_cache()`

Clear all cache

#### Usage

    CacheManager$clear_cache(confirm = FALSE)

#### Arguments

- `confirm`:

  Whether deletion is explicitly confirmed

------------------------------------------------------------------------

### Method `validate_cache()`

Validate cache content

#### Usage

    CacheManager$validate_cache(key)

#### Arguments

- `key`:

  Valid cache key Return deterministic paths for regular RDS cache files
  Resolve a validated cache key to a path inside the cache directory
  Write one R object to a path; separated for deterministic failure
  testing Persist cache data through a same-directory temporary file
  Create stable hash from genes list Create stable hash from models list
  Create stable hash from tissue_name and top_gene_count Create stable
  hash from cluster ID

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    CacheManager$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
