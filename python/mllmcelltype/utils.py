"""Utility functions for LLMCellType."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
from dotenv import dotenv_values

from .config import get_api_key_env_var
from .logger import write_log
from .validation import normalize_text, validate_bool

UNKNOWN_ANNOTATION_TOKENS = {
    "unknown",
    "unk",
    "n/a",
    "na",
    "none",
    "null",
    "nan",
    "-",
    "--",
    "inconclusive",
}
UNKNOWN_WITH_CONTEXT_PATTERN = re.compile(r"(?i)^unknown(?:\s*[\(\[\{].*[\)\]\}])?$")
INCONCLUSIVE_WITH_CONTEXT_PATTERN = re.compile(r"(?i)^inconclusive(?:\s*[\(\[\{].*[\)\]\}])?$")
ERROR_ANNOTATION_PATTERN = re.compile(r"(?i)^error(?:\s*:\s*.*|\s*\(.*\))?$")
CACHE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
CACHE_VERSION = "1.0"
CACHE_ENVELOPE_KEYS = frozenset({"version", "timestamp", "data"})


def is_missing_value(value: Any) -> bool:
    """Return True for scalar missing markers (None/NaN/pandas.NA)."""
    if value is None:
        return True
    with contextlib.suppress(Exception):
        missing = pd.isna(value)
        with contextlib.suppress(Exception):
            return bool(missing)
    return False


def _unwrap_balanced_wrappers(text: str) -> str:
    """Strip one or more balanced wrapper pairs around text."""
    wrapper_pairs = {"[": "]", "(": ")", "{": "}", '"': '"', "'": "'"}
    normalized = text.strip()
    while (
        len(normalized) >= 2
        and normalized[0] in wrapper_pairs
        and normalized[-1] == wrapper_pairs[normalized[0]]
    ):
        normalized = normalized[1:-1].strip()
    return normalized


def is_unknown_annotation(value: Any) -> bool:
    """Whether a value should be treated as missing/unknown annotation."""
    if is_missing_value(value):
        return True

    normalized = _unwrap_balanced_wrappers(str(value))
    if not normalized:
        return True

    lowered = normalized.casefold()
    if lowered in UNKNOWN_ANNOTATION_TOKENS:
        return True
    if UNKNOWN_WITH_CONTEXT_PATTERN.match(normalized):
        return True
    if INCONCLUSIVE_WITH_CONTEXT_PATTERN.match(normalized):
        return True
    return bool(ERROR_ANNOTATION_PATTERN.match(normalized))


def normalize_annotation(value: Any) -> str:
    """Normalize raw annotation into clean text with deterministic Unknown sentinel."""
    if is_unknown_annotation(value):
        return "Unknown"
    return str(value).strip()


def cluster_sort_key(cluster_id: str) -> tuple[int, int, str]:
    """Sort key for deterministic, natural ordering of cluster IDs.

    Numeric IDs sort first by value (0, 1, 2, …, 10, 11);
    non-numeric IDs follow in lexicographic order.
    """
    cluster_str = str(cluster_id)
    try:
        return (0, int(cluster_str), cluster_str)
    except ValueError:
        return (1, 0, cluster_str)


def _get_cache_dir(cache_dir: str | None = None) -> str:
    """Get cache directory path with consistent handling.

    Args:
        cache_dir: Custom cache directory or None for default

    Returns:
        str: Cache directory path
    """
    if cache_dir is None:
        return os.path.join(os.path.expanduser("~"), ".mllmcelltype", "cache")
    return normalize_text(cache_dir, "cache_dir", required=True)


def _get_cache_file(cache_key: str, cache_dir: str | None = None) -> str:
    """Return a cache path while keeping keys inside the cache directory."""
    if not isinstance(cache_key, str) or not CACHE_KEY_PATTERN.fullmatch(cache_key):
        raise ValueError(
            "cache_key must contain 1-128 ASCII letters, digits, underscores, or hyphens"
        )
    return os.path.join(_get_cache_dir(cache_dir), f"{cache_key}.json")


def _list_cache_files(cache_dir: str) -> list[str]:
    """Return deterministic names of regular JSON cache files."""
    return sorted(
        name
        for name in os.listdir(cache_dir)
        if name.endswith(".json") and os.path.isfile(os.path.join(cache_dir, name))
    )


def _validate_cache_age(older_than: int | None) -> int | None:
    """Validate an optional non-negative cache age threshold."""
    if older_than is None:
        return None
    if isinstance(older_than, bool) or not isinstance(older_than, int) or older_than < 0:
        raise ValueError("older_than must be a non-negative integer or None")
    return older_than


def _get_cache_timestamp(cache_data: Any) -> float | None:
    """Return a finite non-negative cache timestamp when metadata is valid."""
    if not isinstance(cache_data, dict) or "timestamp" not in cache_data:
        return None
    timestamp = cache_data["timestamp"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        return None
    timestamp = float(timestamp)
    return timestamp if math.isfinite(timestamp) and timestamp >= 0 else None


def _is_supported_cache_payload(value: Any) -> bool:
    """Return whether a value belongs to the public cache payload contract."""
    return isinstance(value, (list, dict, str))


def _classify_cache_content(cache_content: Any) -> tuple[str, float | None]:
    """Classify loadable cache content and return its optional metadata timestamp."""
    is_envelope = isinstance(cache_content, dict) and CACHE_ENVELOPE_KEYS.issubset(cache_content)
    if is_envelope:
        if set(cache_content) != CACHE_ENVELOPE_KEYS:
            raise ValueError("cache envelope contains unexpected fields")
        if cache_content["version"] != CACHE_VERSION:
            raise ValueError(f"unsupported cache version: {cache_content['version']!r}")
        timestamp = _get_cache_timestamp(cache_content)
        if timestamp is None:
            raise ValueError("cache envelope has an invalid timestamp")
        if not _is_supported_cache_payload(cache_content["data"]):
            raise ValueError("cache envelope has an unsupported data payload")
        return CACHE_VERSION, timestamp

    if not _is_supported_cache_payload(cache_content):
        raise ValueError("legacy cache has an unsupported payload type")
    return "legacy", None


def _api_key_env_vars(provider: str) -> list[str]:
    """Return API key variable names in provider-defined priority order."""
    env_vars = [get_api_key_env_var(provider)]
    if str(provider).strip().lower() == "gemini":
        env_vars.append("GOOGLE_API_KEY")
    return list(dict.fromkeys(env_vars))


def _select_api_key(
    env_vars: list[str],
    values: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Select and normalize the highest-priority configured API key."""
    for env_var in env_vars:
        raw_value = values.get(env_var)
        if isinstance(raw_value, str) and (api_key := raw_value.strip()):
            return api_key, env_var
    return None, None


def _find_dotenv_path() -> str | None:
    """Find the nearest project .env, then the package-level fallback."""
    current_dir = os.path.abspath(os.getcwd())
    for _ in range(4):
        candidate = os.path.join(current_dir, ".env")
        if os.path.isfile(candidate):
            return candidate
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_candidate = os.path.join(package_dir, ".env")
    return package_candidate if os.path.isfile(package_candidate) else None


def load_api_key(provider: str) -> str | None:
    """Load a provider API key from the environment or nearest .env file."""
    env_vars = _api_key_env_vars(provider)
    api_key, selected_env_var = _select_api_key(env_vars, os.environ)
    source = "environment"

    if not api_key and (env_path := _find_dotenv_path()):
        try:
            write_log(f"Found .env file at {env_path}")
            api_key, selected_env_var = _select_api_key(
                env_vars,
                dotenv_values(env_path),
            )
            source = ".env file"
        except (OSError, UnicodeError) as error:
            write_log(f"Error loading .env file: {error!s}", level="warning")

    if api_key:
        write_log(
            f"Using API key from {source} variable: {selected_env_var}",
            level="debug",
        )
    else:
        write_log(f"API key not found for provider: {env_vars[0]}", level="debug")
    return api_key


def _normalize_marker_gene(raw_gene: Any) -> str | None:
    """Normalize one marker gene value or return None when it is missing."""
    if is_missing_value(raw_gene):
        return None
    if isinstance(raw_gene, (bytes, bytearray, memoryview)):
        raise ValueError("individual marker gene values must be text or scalar identifiers")
    if isinstance(raw_gene, Mapping) or (
        isinstance(raw_gene, Iterable) and not isinstance(raw_gene, str)
    ):
        raise ValueError("individual marker gene values must be scalar")
    gene = str(raw_gene).strip()
    return gene or None


def _coerce_marker_gene_list(genes: Any) -> list[str]:
    """Coerce marker genes into a cleaned list of strings."""
    if genes is None:
        return []

    if isinstance(genes, Mapping):
        raise ValueError("marker_genes values must be gene lists, not mapping objects")
    if isinstance(genes, (set, frozenset)):
        raise ValueError("marker_genes values must preserve ranked gene order, not use sets")
    if isinstance(genes, (bytes, bytearray, memoryview)):
        raise ValueError("marker_genes values must contain text or scalar gene identifiers")

    if isinstance(genes, str):
        values = [genes]
    elif isinstance(genes, Iterable):
        values = genes
    else:
        values = [genes]

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_gene in values:
        gene = _normalize_marker_gene(raw_gene)
        if gene and gene not in seen:
            seen.add(gene)
            cleaned.append(gene)
    return cleaned


def _normalize_cluster_id(raw_cluster: Any) -> str | None:
    """Normalize raw cluster identifier to a non-empty string."""
    if is_missing_value(raw_cluster):
        return None
    normalized = str(raw_cluster).strip()
    return normalized or None


def _merge_marker_genes(
    target: dict[str, list[str]],
    normalized_cluster: str,
    genes: list[str],
    raw_cluster: Any,
) -> None:
    """Merge marker genes for the same normalized cluster key."""
    if normalized_cluster not in target:
        target[normalized_cluster] = genes
        return

    existing = target[normalized_cluster]
    existing_set = set(existing)
    appended = 0
    for gene in genes:
        if gene not in existing_set:
            existing.append(gene)
            existing_set.add(gene)
            appended += 1

    write_log(
        f"Cluster key collision after normalization: {raw_cluster!r} -> '{normalized_cluster}', "
        f"merged {appended} marker genes",
        level="warning",
    )


def normalize_marker_genes_keys(
    marker_genes: dict[Any, list[str]],
) -> dict[str, list[str]]:
    """Normalize marker gene dict keys to strings and merge key collisions."""
    if not isinstance(marker_genes, dict):
        raise ValueError(f"marker_genes must be a dict, got {type(marker_genes).__name__}")

    normalized: dict[str, list[str]] = {}
    for raw_cluster, genes in marker_genes.items():
        cluster_id = _normalize_cluster_id(raw_cluster)
        if not cluster_id:
            write_log(
                f"Skipping marker_genes entry with invalid cluster id: {raw_cluster!r}",
                level="warning",
            )
            continue
        cleaned_genes = _coerce_marker_gene_list(genes)
        _merge_marker_genes(normalized, cluster_id, cleaned_genes, raw_cluster)

    return normalized


def create_cache_key(prompt: str, model: str, provider: str, base_url: str | None = None) -> str:
    """Create a cache key for a specific request.

    Args:
        prompt: The prompt text
        model: The model name
        provider: The provider name
        base_url: Optional custom base URL. When provided, it becomes part of
                  the cache key so that different endpoints produce distinct
                  cache entries.

    Returns:
        str: The cache key

    """
    normalized_provider = normalize_text(provider, "provider", required=True).lower()
    normalized_model = normalize_text(model, "model", required=True)
    normalized_prompt = normalize_text(prompt, "prompt", required=True)

    # Create a string to hash with clear separators to avoid collisions
    hash_string = (
        f"provider:{normalized_provider}||model:{normalized_model}||prompt:{normalized_prompt}"
    )

    normalized_base_url = normalize_text(base_url, "base_url")
    normalized_base_url = normalized_base_url.rstrip("/") if normalized_base_url else None
    if normalized_base_url:
        hash_string += f"||base_url:{normalized_base_url}"

    # Create hash
    hash_object = hashlib.sha256(hash_string.encode("utf-8"))
    return hash_object.hexdigest()


def save_to_cache(
    cache_key: str,
    results: list[str] | dict[str, Any] | str,
    cache_dir: str | None = None,
) -> None:
    """Save results to cache.

    Args:
        cache_key: The cache key
        results: The results to cache (list, dictionary, or string)
        cache_dir: The cache directory. If None, uses default directory.

    """
    cache_file = _get_cache_file(cache_key, cache_dir)
    cache_dir = os.path.dirname(cache_file)

    # Ensure results are in a consistent format
    cache_data = {"version": CACHE_VERSION, "timestamp": time.time(), "data": results}

    temp_file: str | None = None
    try:
        if not _is_supported_cache_payload(results):
            raise TypeError("cache results must be a list, dictionary, or string")
        os.makedirs(cache_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_dir,
            prefix=f".{cache_key}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_file = file.name
            json.dump(cache_data, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_file, cache_file)
        write_log(f"Saved results to cache: {cache_file}")
    except (OSError, TypeError, ValueError) as e:
        if temp_file is not None:
            with contextlib.suppress(OSError):
                os.remove(temp_file)
        write_log(f"Error saving cache for {cache_file}: {e}", level="error")


def load_from_cache(
    cache_key: str, cache_dir: str | None = None
) -> list[str] | dict[str, Any] | str | None:
    """Load results from cache.

    Args:
        cache_key: The cache key
        cache_dir: The cache directory. If None, uses default directory.

    Returns:
        list[str] | dict[str, Any] | str | None: Cached results, or None if not found

    """
    cache_file = _get_cache_file(cache_key, cache_dir)

    # Check if cache file exists
    if not os.path.isfile(cache_file):
        return None

    # Load results from cache
    try:
        with open(cache_file) as f:
            cache_content = json.load(f)

        cache_format, _ = _classify_cache_content(cache_content)
        if cache_format == CACHE_VERSION:
            results = cache_content["data"]
            write_log(f"Loaded results from cache (version {CACHE_VERSION}): {cache_file}")
        else:
            # Old format (direct data)
            results = cache_content
            write_log(f"Loaded results from cache (legacy format): {cache_file}")

        return results
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        write_log(f"Error loading cache for {cache_file}: {e}", level="warning")
        return None


def _resolve_marker_dataframe_column(
    marker_genes_df: pd.DataFrame,
    required_name: str,
) -> Any:
    """Resolve one required column without silently accepting normalized duplicates."""
    matches = [
        column for column in marker_genes_df.columns if str(column).strip().lower() == required_name
    ]
    if not matches:
        write_log(
            f"'{required_name}' column not found in marker genes dataframe",
            level="error",
        )
        raise ValueError(
            f"'{required_name}' column not found in marker genes dataframe. "
            f"Available columns: {list(marker_genes_df.columns)}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous '{required_name}' columns after case/whitespace normalization: {matches}"
        )
    return matches[0]


def _append_unique_marker_gene(
    result: dict[str, list[str]],
    gene_sets: dict[str, set[str]],
    cluster_id: str,
    raw_gene: Any,
) -> None:
    """Append one normalized marker gene while preserving first-seen order."""
    gene = _normalize_marker_gene(raw_gene)
    if gene and gene not in gene_sets[cluster_id]:
        gene_sets[cluster_id].add(gene)
        result[cluster_id].append(gene)


def parse_marker_genes(marker_genes_df: pd.DataFrame) -> dict[str, list[str]]:
    """Parse marker genes dataframe into a dictionary.

    Args:
        marker_genes_df: DataFrame containing marker genes

    Returns:
        dict[str, list[str]]: Dictionary mapping cluster names to lists of marker genes

    """
    if not isinstance(marker_genes_df, pd.DataFrame):
        raise ValueError(
            f"marker_genes_df must be a pandas DataFrame, got {type(marker_genes_df).__name__}"
        )

    result = {}

    # Check if dataframe is empty
    if marker_genes_df.empty:
        write_log("Empty marker genes dataframe", level="warning")
        return result

    cluster_col = _resolve_marker_dataframe_column(marker_genes_df, "cluster")
    gene_col = _resolve_marker_dataframe_column(marker_genes_df, "gene")

    # Parse row-wise to avoid hard failure when cluster values are unhashable
    # (e.g., accidental list objects from dirty upstream preprocessing).
    gene_sets: dict[str, set[str]] = {}
    for _, row in marker_genes_df.iterrows():
        raw_cluster = row[cluster_col]
        cluster_id = _normalize_cluster_id(raw_cluster)
        if not cluster_id:
            write_log(
                f"Skipping marker_genes row with invalid cluster id: {raw_cluster!r}",
                level="warning",
            )
            continue

        if cluster_id not in result:
            result[cluster_id] = []
            gene_sets[cluster_id] = set()
        _append_unique_marker_gene(result, gene_sets, cluster_id, row[gene_col])

    return result


LABELED_ANNOTATION_PATTERNS = (
    r"^\s*(?:[-*]\s*)?(.+?)\s*(?::|\uFF1A)\s*(.*?)\s*$",
    r"^\s*(?:[-*]\s*)?(.+?)\s+-\s+(.+?)\s*$",
    r"^\s*(?:[-*]\s*)?([A-Za-z0-9_.-]+)\)\s*(.+?)\s*$",
    r"^\s*(?:[-*]\s*)?([A-Za-z0-9_.-]+)\.\s*(.+?)\s*$",
)


def _parse_labeled_annotation_line(line: str) -> tuple[str, str] | None:
    """Parse an explicitly cluster-labeled annotation line."""
    for pattern in LABELED_ANNOTATION_PATTERNS:
        match = re.match(pattern, line)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    return None


def _build_cluster_aliases(cluster_id: str) -> set[str]:
    """Build accepted prompt-style aliases for one cluster ID."""
    normalized = cluster_id.strip()
    if not normalized:
        return set()

    stripped = re.sub(r"^[Cc]luster[_\s]", "", normalized).strip()
    aliases = {normalized, stripped}
    if stripped:
        aliases.update(
            {
                f"Cluster_{stripped}",
                f"cluster_{stripped}",
                f"Cluster {stripped}",
                f"cluster {stripped}",
            }
        )
    return {alias for alias in aliases if alias}


def _record_unambiguous_alias(
    owners: dict[str, str],
    ambiguous: set[str],
    alias: str,
    cluster_id: str,
) -> None:
    """Record an alias only while it identifies exactly one requested cluster."""
    if alias in ambiguous:
        return
    existing = owners.get(alias)
    if existing is None:
        owners[alias] = cluster_id
    elif existing != cluster_id:
        owners.pop(alias, None)
        ambiguous.add(alias)


def _build_requested_cluster_lookup(clusters: list[str]) -> dict[str, str]:
    """Map exact cluster IDs and unambiguous aliases to requested IDs."""
    exact_ids = list(dict.fromkeys(str(cluster) for cluster in clusters if str(cluster)))
    exact_id_set = set(exact_ids)
    lookup = {cluster_id: cluster_id for cluster_id in exact_ids}
    sensitive_owners: dict[str, str] = {}
    sensitive_ambiguous: set[str] = set()
    insensitive_owners: dict[str, str] = {}
    insensitive_ambiguous: set[str] = set()

    for cluster_id in exact_ids:
        lower_id = cluster_id.lower()
        if lower_id not in exact_id_set:
            _record_unambiguous_alias(
                insensitive_owners,
                insensitive_ambiguous,
                lower_id,
                cluster_id,
            )

        for alias in _build_cluster_aliases(cluster_id):
            if alias == cluster_id or alias in exact_id_set:
                continue
            _record_unambiguous_alias(
                sensitive_owners,
                sensitive_ambiguous,
                alias,
                cluster_id,
            )
            alias_lower = alias.lower()
            if alias_lower not in exact_id_set:
                _record_unambiguous_alias(
                    insensitive_owners,
                    insensitive_ambiguous,
                    alias_lower,
                    cluster_id,
                )

    for alias, cluster_id in sensitive_owners.items():
        lookup.setdefault(alias, cluster_id)
    for alias, cluster_id in insensitive_owners.items():
        lookup.setdefault(alias, cluster_id)
    return lookup


def _resolve_result_cluster(raw_cluster_id: Any, lookup: dict[str, str]) -> str | None:
    """Resolve one result identifier against exact IDs and safe aliases."""
    if raw_cluster_id is None:
        return None
    cluster_id = str(raw_cluster_id).strip()
    if not cluster_id:
        return None

    target = lookup.get(cluster_id) or lookup.get(cluster_id.lower())
    if target:
        return target
    for alias in _build_cluster_aliases(cluster_id):
        target = lookup.get(alias) or lookup.get(alias.lower())
        if target:
            return target
    return None


def _store_preferred_annotation(
    target: dict[str, str],
    cluster_id: str,
    raw_annotation: Any,
) -> None:
    """Store the first annotation, allowing a known label to replace Unknown."""
    annotation = normalize_annotation(raw_annotation)
    existing = target.get(cluster_id)
    if existing is None or (existing == "Unknown" and annotation != "Unknown"):
        target[cluster_id] = annotation


def _resolve_annotation_pairs(
    pairs: Iterable[tuple[Any, Any]],
    cluster_lookup: dict[str, str],
) -> dict[str, str]:
    """Resolve raw cluster/annotation pairs using one shared conflict policy."""
    resolved: dict[str, str] = {}
    for raw_cluster_id, raw_annotation in pairs:
        cluster_id = _resolve_result_cluster(raw_cluster_id, cluster_lookup)
        if cluster_id:
            _store_preferred_annotation(resolved, cluster_id, raw_annotation)
    return resolved


def _complete_cluster_annotations(
    annotations: dict[str, str],
    clusters: list[str],
) -> dict[str, str]:
    """Return all requested cluster IDs with a canonical Unknown fallback."""
    return {str(cluster): annotations.get(str(cluster), "Unknown") for cluster in clusters}


def _normalize_result_lines(results: Any) -> list[str]:
    """Normalize provider output into deterministic non-empty lines."""
    if isinstance(results, str):
        raw_lines = results.splitlines()
    elif isinstance(results, set):
        raw_lines = sorted(str(line) for line in results)
    elif isinstance(results, (list, tuple)):
        raw_lines = [str(line) for line in results]
    elif results is None:
        raw_lines = []
    else:
        raw_lines = [str(results)]
    return [line.strip() for line in raw_lines if line.strip()]


def _extract_json_document(lines: list[str]) -> Any | None:
    """Decode the first JSON document represented by response lines."""
    full_text = "\n".join(lines)
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", full_text)
    if code_block:
        json_text = code_block.group(1)
    else:
        document = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", full_text)
        json_text = document.group(1) if document else full_text

    try:
        return json.loads(json_text)
    except (json.JSONDecodeError, TypeError) as error:
        write_log(f"Failed to parse JSON response: {error!s}", level="debug")
        return None


def _select_json_annotation(record: dict[str, Any]) -> Any:
    """Select the first populated annotation field from a JSON record."""
    for field in ("cell_type", "annotation", "label"):
        value = record.get(field)
        if not is_missing_value(value) and str(value).strip():
            return value
    return None


def _json_annotation_pairs(payload: Any) -> list[tuple[Any, Any]]:
    """Convert supported JSON response shapes into cluster/annotation pairs."""
    if isinstance(payload, dict) and isinstance(payload.get("annotations"), list):
        records = payload["annotations"]
    elif isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        return list(payload.items())
    else:
        return []

    return [
        (record["cluster"], _select_json_annotation(record))
        for record in records
        if isinstance(record, dict) and "cluster" in record
    ]


def _looks_like_cluster_ref(key: str) -> bool:
    """Whether a label key denotes a cluster id (bare number or 'Cluster N')."""
    stripped = re.sub(r"(?i)^cluster[\s_]*", "", key.strip())
    return stripped.isdigit()


def _parse_labeled_annotations(
    lines: list[str],
    cluster_lookup: dict[str, str],
    expected_count: int,
) -> tuple[dict[str, str], bool]:
    """Parse explicitly cluster-labeled lines.

    Returns ``(labeled, is_labeled_response)``. ``labeled`` maps each resolved
    cluster ID to its annotation (keyed by cluster ID, so out-of-order labels
    are placed correctly). The response is treated as labeled when at least one
    label resolves to a requested cluster, UNLESS an unresolved cluster-reference
    key (an ordinal or ``Cluster N``) coexists with incomplete coverage — that
    signals list ordinals misaligned against the requested clusters (e.g. a
    1-based numbered list on 0-based clusters), which must fall through to
    positional mapping.

    A line counts only when it parses into a key/value pair with a non-empty
    value; a bare ``"Header:"`` (empty value) is a preamble. A stray word-keyed
    colon line (``Summary:``, ``Neurons: excitatory``) is ignored, and a
    hallucinated out-of-range label (``5: X`` when clusters are 0..2) does not
    spoil an otherwise fully-covered labeled response.
    """
    labeled: dict[str, str] = {}
    resolved_clusters: set[str] = set()
    has_unresolved_cluster_ref = False
    for line in lines:
        parsed = _parse_labeled_annotation_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if not value:
            continue  # preamble line such as "Here are the annotations:"
        cluster_id = _resolve_result_cluster(key, cluster_lookup)
        if cluster_id is not None:
            resolved_clusters.add(cluster_id)
            _store_preferred_annotation(labeled, cluster_id, value)
        elif _looks_like_cluster_ref(key):
            has_unresolved_cluster_ref = True
        # else: a stray word-keyed colon line (Summary:, Neurons: excitatory)
    is_labeled_response = bool(resolved_clusters) and not (
        has_unresolved_cluster_ref and len(resolved_clusters) < expected_count
    )
    return labeled, is_labeled_response


def _extract_positional_annotation(
    line: str,
    cluster_lookup: dict[str, str],
) -> str | None:
    """Extract one line's annotation content for positional mapping.

    Returns None only for lines that do not occupy a cluster slot: an
    empty-value preamble/header (``"Notes:"``) and an explicit label for an
    unrequested/nonexistent cluster (``"Cluster 999: Noise"``). Every other
    line — including a sentinel such as ``"Unknown"`` — keeps its slot so the
    line->cluster correspondence is preserved (the sentinel maps to Unknown).
    A numbered-list ordinal (``"1. T cells"``) or a resolvable label prefix
    (``"0: T cells"``) is stripped to its content; a colon that is part of the
    annotation (``"Neurons: excitatory"``) is kept whole.
    """
    numbered = re.match(r"^\s*(?:[-*]\s*)?\d+\s*[.)]\s+(.+?)\s*$", line)
    if numbered is not None:
        return numbered.group(1)

    parsed = _parse_labeled_annotation_line(line)
    if parsed is not None:
        key, value = parsed
        if not value:
            return None  # preamble/header line such as "Here are the annotations:"
        if _resolve_result_cluster(key, cluster_lookup):
            return value
        if _looks_like_cluster_ref(key) and (":" in line or "：" in line):  # noqa: RUF001
            return None  # explicit colon label for an unrequested/nonexistent cluster
        # otherwise part of the annotation ("Neurons: excitatory", "5 - HT neurons")
    return line


def format_results(
    results: list[str] | dict[str, Any] | str,
    clusters: list[str],
) -> dict[str, str]:
    """Format provider results into a complete cluster-to-annotation mapping."""
    cluster_lookup = _build_requested_cluster_lookup(clusters)

    if isinstance(results, dict):
        resolved = _resolve_annotation_pairs(results.items(), cluster_lookup)
        write_log("Parsed response from dictionary format", level="info")
        return _complete_cluster_annotations(resolved, clusters)

    lines = _normalize_result_lines(results)
    expected_count = len(set(map(str, clusters)))

    labeled, is_labeled_response = _parse_labeled_annotations(
        lines, cluster_lookup, expected_count
    )
    if is_labeled_response:
        if len(labeled) == expected_count:
            write_log("Successfully parsed labeled response", level="info")
        else:
            write_log(
                f"Parsed partial labeled response ({len(labeled)}/{expected_count} clusters), "
                "missing clusters marked Unknown",
                level="warning",
            )
        return _complete_cluster_annotations(labeled, clusters)

    json_payload = _extract_json_document(lines)
    json_annotations = _resolve_annotation_pairs(
        _json_annotation_pairs(json_payload),
        cluster_lookup,
    )
    if json_annotations:
        if len(json_annotations) == expected_count:
            write_log("Successfully parsed JSON response", level="info")
        else:
            write_log(
                f"Parsed partial JSON response ({len(json_annotations)}/{len(clusters)} clusters), "
                "missing clusters marked Unknown",
                level="warning",
            )
        return _complete_cluster_annotations(json_annotations, clusters)

    # Positional fallback: strip ordinal/label prefixes, drop preamble and
    # non-annotation lines, then map surviving annotations onto clusters in order.
    contents = [
        content
        for content in (
            _extract_positional_annotation(line, cluster_lookup) for line in lines
        )
        if content is not None
    ]
    if len(contents) < len(clusters):
        write_log(
            f"Fewer annotation lines ({len(contents)}) than clusters ({len(clusters)}), "
            "remaining clusters will be marked Unknown",
            level="warning",
        )
    elif len(contents) > len(clusters):
        write_log(
            f"More annotation lines ({len(contents)}) than clusters ({len(clusters)}), "
            "extra lines will be ignored",
            level="warning",
        )

    positional = {
        str(cluster): normalize_annotation(contents[index])
        for index, cluster in enumerate(clusters)
        if index < len(contents)
    }
    write_log("Parsed response as line-by-line format", level="info")
    return _complete_cluster_annotations(positional, clusters)


def clear_cache(cache_dir: str | None = None, older_than: int | None = None) -> int:
    """Clear cache.

    Args:
        cache_dir: Cache directory
        older_than: Only clear items older than this many seconds.
                   If None, clear all cache.

    Returns:
        int: Number of cache files removed

    """
    cache_dir = _get_cache_dir(cache_dir)
    older_than = _validate_cache_age(older_than)

    if not os.path.exists(cache_dir):
        return 0

    # Get all cache files
    cache_files = _list_cache_files(cache_dir)

    if older_than is None:
        # Remove all cache files
        count = 0
        for f in cache_files:
            try:
                os.remove(os.path.join(cache_dir, f))
                count += 1
            except OSError as e:
                write_log(f"Error removing cache file {f}: {e}", level="warning")
        return count
    # Remove only older files
    now = time.time()
    count = 0
    for f in cache_files:
        file_path = os.path.join(cache_dir, f)
        try:
            # Use envelope metadata only after applying the same schema validation
            # as normal cache reads. Legacy and invalid entries fall back to mtime.
            try:
                with open(file_path) as file:
                    cache_data = json.load(file)
                _, timestamp = _classify_cache_content(cache_data)
                file_age = (
                    now - timestamp if timestamp is not None else now - os.path.getmtime(file_path)
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                file_age = now - os.path.getmtime(file_path)
                write_log(
                    f"Invalid cache metadata in {f}; falling back to file mtime",
                    level="warning",
                )
            if file_age > older_than:
                os.remove(file_path)
                count += 1
        except OSError as e:
            write_log(f"Error processing cache file {f}: {e}", level="warning")

    return count


def get_cache_stats(cache_dir: str | None = None, detailed: bool = True) -> dict[str, Any]:
    """Get cache statistics.

    Args:
        cache_dir: The cache directory (None for default)
        detailed: If True, read file contents for detailed stats (slower).
                  If False, only return basic file counts and sizes (faster).

    Returns:
        dict[str, Any]: Cache statistics

    """
    detailed = validate_bool(detailed, "detailed")
    cache_dir = _get_cache_dir(cache_dir)

    if not os.path.exists(cache_dir):
        result = {
            "exists": False,
            "path": cache_dir,
            "count": 0,
            "size": 0,
            "size_mb": 0.0,
        }
        if detailed:
            result.update(
                {
                    "status": "No cache directory",
                    "oldest": None,
                    "newest": None,
                    "format_counts": {
                        "legacy": 0,
                        CACHE_VERSION: 0,
                        "unknown": 0,
                    },
                    "valid_files": 0,
                    "invalid_files": 0,
                }
            )
        return result

    # Get all cache files
    cache_files = _list_cache_files(cache_dir)
    total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cache_files)

    # Basic stats (fast path)
    result = {
        "exists": True,
        "path": cache_dir,
        "count": len(cache_files),
        "size": total_size,
        "size_mb": total_size / (1024 * 1024),
    }

    if not detailed:
        return result

    # Detailed stats (slow path - reads file contents)
    if not cache_files:
        result.update(
            {
                "status": "Empty cache",
                "oldest": None,
                "newest": None,
                "format_counts": {"legacy": 0, CACHE_VERSION: 0, "unknown": 0},
                "valid_files": 0,
                "invalid_files": 0,
            }
        )
        return result

    oldest = float("inf")
    newest = 0
    format_counts = {"legacy": 0, CACHE_VERSION: 0, "unknown": 0}
    valid_files = 0
    invalid_files = 0

    for f in cache_files:
        file_path = os.path.join(cache_dir, f)
        try:
            # Load cache data
            with open(file_path) as file:
                cache_data = json.load(file)

            cache_format, timestamp = _classify_cache_content(cache_data)
            valid_files += 1
            format_counts[cache_format] += 1
            if timestamp is not None:
                oldest = min(oldest, timestamp)
                newest = max(newest, timestamp)

        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
        ) as e:
            invalid_files += 1
            format_counts["unknown"] += 1
            write_log(f"Error processing cache file {f}: {e}", level="warning")

    # Convert timestamps to readable format
    oldest_str = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(oldest))
        if oldest != float("inf")
        else None
    )
    newest_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest)) if newest != 0 else None

    result.update(
        {
            "status": "Cache available",
            "valid_files": valid_files,
            "invalid_files": invalid_files,
            "oldest": oldest_str,
            "newest": newest_str,
            "format_counts": format_counts,
        }
    )

    return result
