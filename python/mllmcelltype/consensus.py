"""Module for consensus annotation of cell types from multiple LLM predictions."""

from __future__ import annotations

import contextlib
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from .annotate import annotate_clusters, get_model_response
from .config import (
    DEFAULT_FALLBACK_CONSENSUS_PROPORTION,
    DEFAULT_FALLBACK_ENTROPY,
    get_default_model,
    get_supported_providers,
)
from .execution import annotation_execution, current_execution
from .functions import get_provider, validate_provider_model_match
from .logger import write_log
from .prompts import (
    create_cell_type_extraction_prompt,
    create_consensus_check_prompt,
    create_discussion_consensus_prompt,
    create_discussion_prompt,
    create_initial_discussion_prompt,
    validate_prompt_template,
)
from .url_utils import resolve_provider_base_url
from .utils import (
    cluster_sort_key,
    is_missing_value,
    is_unknown_annotation,
    load_api_key,
    normalize_annotation,
    normalize_marker_genes_keys,
)
from .validation import normalize_text, validate_bool

# Default result structure for discussion round consensus check
# Used when consensus check fails or has insufficient data
DEFAULT_CONSENSUS_RESULT = {
    "reached": False,
    "consensus_proportion": DEFAULT_FALLBACK_CONSENSUS_PROPORTION,
    "entropy": DEFAULT_FALLBACK_ENTROPY,
    "majority_prediction": "Unknown",
}

# Recoverable errors from provider calls and response parsing.
# These should not crash full consensus workflows.
RECOVERABLE_LLM_EXCEPTIONS = (
    requests.RequestException,
    ValueError,
    KeyError,
    json.JSONDecodeError,
    AttributeError,
    ImportError,
    TypeError,
    RuntimeError,
)

ModelSpecInput = str | dict[str, str]


@dataclass(frozen=True)
class _ResolvedModelSpec:
    """Canonical provider/model identity used by consensus internals."""

    provider: str
    model: str

    @property
    def key(self) -> str:
        """Return the collision-safe identity used in result mappings."""
        return f"{self.provider}:{self.model}"


def _normalize_probability(value: Any, field_name: str) -> float:
    """Validate a finite numeric probability without accepting booleans."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{field_name} must be a finite number between 0 and 1")
    return float(value)


def _normalize_nonnegative_number(value: Any, field_name: str) -> float:
    """Validate a finite non-negative numeric control."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return float(value)


def _normalize_nonnegative_integer(value: Any, field_name: str) -> int:
    """Validate a non-negative integer control."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _normalize_api_keys(api_keys: dict[str, str] | None) -> dict[str, str]:
    """Normalize and validate provider API keys.

    Rules:
    - provider keys are normalized to lowercase strings
    - key values must be strings
    - blank keys are treated as missing and dropped
    """
    if api_keys is None:
        return {}
    if not isinstance(api_keys, dict):
        raise ValueError(
            f"api_keys must be a dict mapping provider names to keys, got {type(api_keys).__name__}"
        )

    normalized: dict[str, str] = {}
    for raw_provider, raw_key in api_keys.items():
        if not isinstance(raw_provider, str):
            raise ValueError(
                f"API key provider names must be strings, got {type(raw_provider).__name__}"
            )
        provider = raw_provider.strip().lower()
        if not provider:
            raise ValueError("API key provider names must be non-empty strings")
        if provider in normalized:
            raise ValueError(
                f"Duplicate API key provider after case/whitespace normalization: '{provider}'"
            )
        if raw_key is None:
            continue
        if not isinstance(raw_key, str):
            raise ValueError(
                f"API key for provider '{provider}' must be a string, got {type(raw_key).__name__}"
            )
        key = raw_key.strip()
        if key:
            normalized[provider] = key

    return normalized


def _normalize_consensus_model_spec(
    consensus_model: str | dict[str, str] | None,
) -> dict[str, str] | None:
    """Normalize and validate consensus_model specification."""
    if consensus_model is None:
        return None

    if isinstance(consensus_model, str):
        model_name = consensus_model.strip()
        if not model_name:
            raise ValueError("consensus_model string cannot be empty")
        provider = get_provider(model_name)
        return {"provider": provider, "model": model_name}

    if not isinstance(consensus_model, dict):
        raise ValueError(
            f"consensus_model must be a string or dict, got {type(consensus_model).__name__}"
        )

    provider_normalized = _normalize_consensus_model_provider_field(consensus_model.get("provider"))
    model_normalized = _normalize_consensus_model_name_field(consensus_model.get("model"))

    if provider_normalized is None and model_normalized is None:
        raise ValueError("consensus_model must include at least one of 'provider' or 'model'")

    if provider_normalized is None and model_normalized is not None:
        provider_normalized = get_provider(model_normalized)
    if provider_normalized is not None and model_normalized is None:
        model_normalized = get_default_model(provider_normalized)

    validate_provider_model_match(provider_normalized, model_normalized, "consensus_model")

    return {"provider": provider_normalized, "model": model_normalized}


def _normalize_model_identity(raw_model: Any, field_name: str) -> str:
    """Normalize one model identity while rejecting ambiguous coercion."""
    if not isinstance(raw_model, str):
        raise ValueError(f"{field_name} must be a non-empty string")
    model_name = raw_model.strip()
    if not model_name:
        raise ValueError(f"{field_name} must be a non-empty string")
    return model_name


def _normalize_consensus_model_provider_field(provider: Any) -> str | None:
    """Normalize optional consensus provider field."""
    if provider is None:
        return None
    if not isinstance(provider, str):
        raise ValueError(
            f"consensus_model.provider must be a string when provided, got {type(provider).__name__}"
        )
    normalized = provider.strip().lower()
    if not normalized:
        raise ValueError("consensus_model.provider cannot be empty when provided")
    if normalized not in get_supported_providers():
        raise ValueError(f"Unsupported consensus_model.provider: {normalized}")
    return normalized


def _normalize_consensus_model_name_field(model: Any) -> str | None:
    """Normalize optional consensus model field."""
    if model is None:
        return None
    if not isinstance(model, str):
        raise ValueError(
            f"consensus_model.model must be a string when provided, got {type(model).__name__}"
        )
    normalized = model.strip()
    if not normalized:
        raise ValueError("consensus_model.model cannot be empty when provided")
    return normalized


def _normalize_single_prediction_map(model_name: str, raw_results: Any) -> dict[str, str]:
    """Normalize one model's cluster->annotation payload."""
    if raw_results is None:
        write_log(f"Model '{model_name}' returned no predictions, skipping", level="warning")
        return {}

    if not isinstance(raw_results, dict):
        write_log(
            f"Model '{model_name}' predictions must be a dict, got {type(raw_results).__name__}; skipping",
            level="warning",
        )
        return {}

    cluster_map: dict[str, str] = {}
    for raw_cluster, raw_annotation in raw_results.items():
        if is_missing_value(raw_cluster) or is_missing_value(raw_annotation):
            continue
        cluster_id = str(raw_cluster).strip()
        if not cluster_id:
            continue
        if not isinstance(raw_annotation, str):
            raise ValueError(
                f"Model '{model_name}' annotation for cluster '{cluster_id}' must be a string"
            )
        annotation = raw_annotation.strip()
        if annotation and not is_unknown_annotation(annotation):
            if cluster_id in cluster_map:
                raise ValueError(
                    f"Model '{model_name}' has duplicate cluster ID after normalization: "
                    f"'{cluster_id}'"
                )
            cluster_map[cluster_id] = annotation
    return cluster_map


def _normalize_predictions(
    predictions: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Normalize consensus prediction payload with defensive validation."""
    if not isinstance(predictions, dict):
        raise ValueError(
            f"predictions must be a dict mapping model->cluster annotations, got {type(predictions).__name__}"
        )

    normalized: dict[str, dict[str, str]] = {}
    seen_model_names: set[str] = set()
    for raw_model, raw_results in predictions.items():
        model_name = _normalize_model_identity(raw_model, "prediction model names")
        if model_name in seen_model_names:
            raise ValueError(
                f"Duplicate prediction model name after whitespace normalization: '{model_name}'"
            )
        seen_model_names.add(model_name)
        cluster_map = _normalize_single_prediction_map(model_name, raw_results)

        if cluster_map:
            normalized[model_name] = cluster_map
        else:
            write_log(f"Model '{model_name}' had no valid annotations, skipping", level="warning")

    return normalized


def _collect_valid_round_responses(round_responses: dict[str, Any]) -> dict[str, str]:
    """Normalize round responses and drop provider error outputs."""
    if not isinstance(round_responses, dict):
        raise ValueError("round_responses must be a dict mapping model name to response text")

    valid_responses: dict[str, str] = {}
    seen_model_names: set[str] = set()
    for raw_model_name, response in round_responses.items():
        model_name = _normalize_model_identity(raw_model_name, "round response model names")
        if model_name in seen_model_names:
            raise ValueError(
                f"Duplicate round response model name after whitespace normalization: '{model_name}'"
            )
        seen_model_names.add(model_name)
        if is_missing_value(response):
            continue
        if not isinstance(response, str):
            raise ValueError(f"Round response from model '{model_name}' must be a string")
        response_text = response.strip()
        if not response_text or response_text.casefold().startswith("error:"):
            continue
        if is_unknown_annotation(response_text):
            continue
        valid_responses[model_name] = response_text
    return valid_responses


def _build_metadata(
    *,
    models: list[str | dict[str, str]],
    species: str,
    tissue: str | None,
    consensus_threshold: float,
    entropy_threshold: float,
    max_discussion_rounds: int,
) -> dict[str, Any]:
    """Build metadata payload for consensus outputs."""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
        "species": species,
        "tissue": tissue,
        "consensus_threshold": consensus_threshold,
        "entropy_threshold": entropy_threshold,
        "max_discussion_rounds": max_discussion_rounds,
    }


def _build_interactive_result(
    *,
    metadata: dict[str, Any],
    consensus: dict[str, str] | None = None,
    consensus_proportion: dict[str, float] | None = None,
    entropy: dict[str, float] | None = None,
    controversial_clusters: list[str] | None = None,
    resolved: dict[str, str] | None = None,
    model_annotations: dict[str, dict[str, str]] | None = None,
    discussion_logs: dict[str, list[dict]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build normalized return payload for interactive consensus."""
    normalized_discussion_logs = discussion_logs or {}
    normalized_controversial_clusters = [
        str(cluster_id) for cluster_id in (controversial_clusters or [])
    ]
    discussion_round_counts = {cluster_id: 0 for cluster_id in normalized_controversial_clusters}
    discussion_round_counts.update(
        {
            str(cluster_id): len(rounds) if isinstance(rounds, list) else 0
            for cluster_id, rounds in normalized_discussion_logs.items()
        }
    )

    result = {
        "consensus": consensus or {},
        "consensus_proportion": consensus_proportion or {},
        "entropy": entropy or {},
        "controversial_clusters": normalized_controversial_clusters,
        "resolved": resolved or {},
        "model_annotations": model_annotations or {},
        "discussion_logs": normalized_discussion_logs,
        "discussion_round_counts": discussion_round_counts,
        "metadata": metadata,
    }
    if error:
        result["error"] = error
    return result


def _resolve_api_keys_for_models(
    models: list[_ResolvedModelSpec],
    api_keys: dict[str, str] | None,
) -> dict[str, str]:
    """Resolve API keys for all model providers.

    Priority per provider:
    1) caller-provided api_keys entry
    2) environment/.env via load_api_key(provider)
    """
    resolved_keys = _normalize_api_keys(api_keys)

    for model_spec in models:
        if resolved_keys.get(model_spec.provider):
            continue
        env_key = load_api_key(model_spec.provider)
        if env_key:
            resolved_keys[model_spec.provider] = env_key

    return resolved_keys


def _normalize_models_spec(models: Any) -> list[_ResolvedModelSpec]:
    """Parse public model inputs into one canonical internal representation."""
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list of model specifications")

    supported_providers = set(get_supported_providers())
    normalized: list[_ResolvedModelSpec] = []

    for index, item in enumerate(models):
        if isinstance(item, _ResolvedModelSpec):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(_normalize_model_spec_dict(item, index, supported_providers))
        else:
            normalized.append(_normalize_model_spec_scalar(item, index))

    return normalized


def _normalize_model_spec_dict(
    item: dict[str, str],
    index: int,
    supported_providers: set[str],
) -> _ResolvedModelSpec:
    """Normalize one dict-form model specification."""
    model_name = item.get("model")
    provider_name = item.get("provider")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            f"Invalid model specification at index {index}: "
            f"dict must include a non-empty string 'model' key, got {item}"
        )
    model = model_name.strip()

    if provider_name is None:
        provider = get_provider(model)
    elif not isinstance(provider_name, str):
        raise ValueError(
            f"Invalid model specification at index {index}: "
            f"'provider' must be a string when provided, got {item}"
        )
    else:
        provider = provider_name.strip().lower()
        if not provider:
            raise ValueError(
                f"Invalid model specification at index {index}: "
                f"'provider' must be a non-empty string when provided, got {item}"
            )
        if provider not in supported_providers:
            raise ValueError(
                f"Invalid model specification at index {index}: "
                f"unsupported provider '{provider_name}'. "
                f"Supported providers: {sorted(supported_providers)}"
            )

    validate_provider_model_match(
        provider,
        model,
        f"models[{index}]",
    )
    return _ResolvedModelSpec(provider=provider, model=model)


def _normalize_model_spec_scalar(item: Any, index: int) -> _ResolvedModelSpec:
    """Normalize one scalar-form model specification."""
    if isinstance(item, str):
        model = item.strip()
        if model:
            return _ResolvedModelSpec(provider=get_provider(model), model=model)
    elif item:
        raise ValueError(
            f"Invalid model specification at index {index}: "
            f"expected string or dict, got {type(item).__name__}"
        )

    raise ValueError(
        f"Invalid model specification at index {index}: model name must be a non-empty string"
    )


def _iter_supported_provider_keys(api_keys: dict[str, str]) -> list[tuple[str, str, str]]:
    """Return (provider, default_model, api_key) for supported configured providers only."""
    supported: list[tuple[str, str, str]] = []
    supported_providers = set(get_supported_providers())
    for provider, key in _normalize_api_keys(api_keys).items():
        if not key:
            continue
        if provider not in supported_providers:
            write_log(
                f"Ignoring unsupported provider key in consensus flow: {provider}",
                level="debug",
            )
            continue
        default_model = get_default_model(provider)
        supported.append((provider, default_model, key))
    return supported


def _has_supported_api_key(api_keys: dict[str, str]) -> bool:
    """Whether at least one supported provider key is configured."""
    return bool(_iter_supported_provider_keys(api_keys))


def _resolve_consensus_provider(
    consensus_model: dict[str, str] | None,
    api_keys: dict[str, str],
) -> tuple[str | None, str | None, str | None]:
    """Resolve provider, model, and API key for consensus checking.

    Resolution order: explicit ``consensus_model`` dict → first available
    key in ``api_keys``.

    Args:
        consensus_model: Optional dict with 'provider' and/or 'model' keys
        api_keys: Dictionary mapping provider names to API keys

    Returns:
        tuple of (provider, model, api_key) — any element may be None
    """
    normalized_api_keys = _normalize_api_keys(api_keys)
    supported_keys = _iter_supported_provider_keys(normalized_api_keys)

    if consensus_model:
        if not isinstance(consensus_model, dict):
            raise ValueError(
                f"consensus_model must be a dict when passed to _resolve_consensus_provider, got {type(consensus_model).__name__}"
            )
        provider = consensus_model.get("provider")
        if isinstance(provider, str):
            provider = provider.strip().lower()
            if not provider:
                provider = None
        model = consensus_model.get("model")
        if isinstance(model, str):
            model = model.strip()
            if not model:
                model = None
        if not provider and model:
            provider = get_provider(model)
        if provider and not model:
            model = get_default_model(provider)
    else:
        provider = None
        model = None
        if supported_keys:
            provider, model, _key = supported_keys[0]

    api_key = normalized_api_keys.get(provider) if provider else None
    return provider, model, api_key


def _resolve_fallback_target(
    *,
    provider: str | None,
    fallback_provider: str | None,
    fallback_model: str | None,
    supported_keys: list[tuple[str, str, str]],
) -> tuple[str | None, str | None]:
    """Resolve fallback provider/model target from explicit args or configured keys."""
    if fallback_provider is None:
        for candidate_provider, default_model, _key in supported_keys:
            if candidate_provider != provider:
                return candidate_provider, default_model
        return None, None

    supported_provider_names = {provider_name for provider_name, _model, _key in supported_keys}
    if fallback_provider not in supported_provider_names:
        write_log(
            f"Ignoring unsupported fallback provider in consensus flow: {fallback_provider}",
            level="debug",
        )
        return None, None

    if fallback_model:
        return fallback_provider, fallback_model

    with contextlib.suppress(ValueError):
        return fallback_provider, get_default_model(fallback_provider)
    return fallback_provider, fallback_model


def _call_primary_provider_once(
    *,
    prompt: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    base_urls: str | dict[str, str] | None,
) -> str | None:
    """Call one provider once; provider adapters own transport retries."""
    if not api_key:
        write_log(f"No API key for {provider}, trying fallback", level="debug")
        return None

    primary_base_url = resolve_provider_base_url(provider, base_urls)
    try:
        response = get_model_response(
            prompt=prompt,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=primary_base_url,
            use_cache=current_execution().use_cache if current_execution() else True,
        )
        write_log(f"Successfully got response from {provider}")
        return response
    except RECOVERABLE_LLM_EXCEPTIONS as error:
        write_log(f"Provider {provider} request failed: {error!s}", level="warning")
        return None


def _call_fallback_provider_once(
    *,
    prompt: str,
    fallback_provider: str | None,
    fallback_model: str | None,
    normalized_api_keys: dict[str, str],
    base_urls: str | dict[str, str] | None,
) -> str | None:
    """Call fallback provider once when configured."""
    if not fallback_provider or not fallback_model:
        return None

    fallback_api_key = normalized_api_keys.get(fallback_provider)
    if not fallback_api_key:
        return None

    fallback_base_url = resolve_provider_base_url(fallback_provider, base_urls)
    try:
        response = get_model_response(
            prompt=prompt,
            provider=fallback_provider,
            model=fallback_model,
            api_key=fallback_api_key,
            base_url=fallback_base_url,
        )
        write_log(f"Successfully got response from {fallback_provider} as fallback")
        return response
    except RECOVERABLE_LLM_EXCEPTIONS as e:
        write_log(f"Error on {fallback_provider} fallback: {e!s}", level="warning")
        return None


def _call_llm_with_fallback(
    prompt: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
    api_keys: dict[str, str] | None = None,
    base_urls: str | dict[str, str] | None = None,
) -> str | None:
    """Call a primary LLM once, then one configured fallback.

    API key resolution: explicit ``api_key`` arg → ``api_keys`` dict.
    This function does NOT load keys from environment variables or
    ``.env`` files; the caller (or the entry-point function) is
    responsible for populating ``api_keys`` with all available keys.

    Args:
        prompt: The prompt to send
        provider: Primary provider to use
        model: Primary model to use
        api_key: API key for primary provider (falls back to api_keys dict)
        fallback_provider: Fallback provider if primary fails
        fallback_model: Fallback model if primary fails
        api_keys: Dictionary of API keys keyed by provider name
        base_urls: Custom base URLs for API endpoints

    Returns:
        str | None: LLM response or None if all attempts failed
    """
    normalized_api_keys = _normalize_api_keys(api_keys)
    supported_keys = _iter_supported_provider_keys(normalized_api_keys)
    provider = provider.lower() if provider else None
    fallback_provider = fallback_provider.lower() if fallback_provider else None

    # Resolve primary API key from explicit arg or api_keys dict.
    # We intentionally do NOT call load_api_key() here: if the caller
    # didn't provide the key, it means the provider is not configured.
    if not api_key:
        api_key = normalized_api_keys.get(provider) if provider else None

    if (run := current_execution()) is not None:
        # Keep fallback within the models selected and validated for this run.
        candidates = [(provider, model), *run.consensus_candidates]
        attempted = set()
        for candidate_provider, candidate_model in candidates:
            key = f"{candidate_provider}:{candidate_model}"
            if key in attempted or key in run.unavailable:
                continue
            attempted.add(key)
            response = _call_primary_provider_once(
                prompt=prompt,
                provider=candidate_provider,
                model=candidate_model,
                api_key=api_key
                if (candidate_provider, candidate_model) == (provider, model)
                else normalized_api_keys.get(candidate_provider),
                base_urls=base_urls,
            )
            if response is not None:
                return response
            if len(attempted) >= 2:
                break
        return None

    fallback_provider, fallback_model = _resolve_fallback_target(
        provider=provider,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        supported_keys=supported_keys,
    )

    primary_response = _call_primary_provider_once(
        prompt=prompt,
        provider=provider,
        model=model,
        api_key=api_key,
        base_urls=base_urls,
    )
    if primary_response is not None:
        return primary_response

    fallback_response = _call_fallback_provider_once(
        prompt=prompt,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        normalized_api_keys=normalized_api_keys,
        base_urls=base_urls,
    )
    if fallback_response is not None:
        return fallback_response

    write_log(f"All LLM attempts failed for provider {provider}", level="warning")

    return None


def _normalize_predicted_label(label: str | None) -> str | None:
    """Normalize parsed label and drop empty/Unknown sentinels."""
    if label is None:
        return None
    normalized = str(label).strip()
    if len(normalized) >= 2 and normalized[0] == "[" and normalized[-1] == "]":
        normalized = normalized[1:-1].strip()
    return normalized if normalized and not is_unknown_annotation(normalized) else None


def _parse_standard_metrics_lines(
    lines: list[str],
) -> tuple[float | None, float | None, str | None]:
    """Parse strict 4-line [0/1, CP, H, annotation] format."""
    if len(lines) < 4:
        return None, None, None

    result_lines = lines[-4:]
    if not re.match(r"^\s*[01]\s*$", result_lines[0]):
        return None, None, None
    if not re.match(r"^\s*(0\.\d+|1\.0*|[01])\s*$", result_lines[1]):
        return None, None, None
    if not re.match(r"^\s*(\d+\.\d+|\d+)\s*$", result_lines[2]):
        return None, None, None

    with contextlib.suppress(ValueError, IndexError):
        cp_value = float(result_lines[1].strip())
        h_value = float(result_lines[2].strip())
        annotation = _normalize_predicted_label(result_lines[3])
        write_log("Detected standard 4-line format", level="debug")
        return cp_value, h_value, annotation

    return None, None, None


def _extract_consensus_indicator(text: str) -> bool | None:
    """Extract an explicit consensus decision from supported response formats."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4 and re.fullmatch(r"[01]", lines[-4]):
        return lines[-4] == "1"

    json_text = _extract_json_block_from_text(text)
    if json_text:
        with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
            payload = json.loads(json_text)
            if isinstance(payload, dict):
                for key in ("reached", "consensus", "consensus_reached", "consensusReached"):
                    value = payload.get(key)
                    if isinstance(value, bool):
                        return value
                    if value in (0, 1, "0", "1"):
                        return str(value) == "1"

    labeled_match = re.search(
        r"(?im)^\s*consensus(?:\s+reached)?\s*[:=]\s*(0|1|true|false|yes|no)\s*$",
        text,
    )
    if labeled_match:
        return labeled_match.group(1).casefold() in {"1", "true", "yes"}
    return None


def _extract_json_block_from_text(text: str) -> str | None:
    """Extract likely JSON payload from free-text response."""
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block_match:
        return code_block_match.group(1)

    brace_match = re.search(r"(\{[\s\S]*\})", text)
    if brace_match:
        return brace_match.group(1)
    return None


def _extract_first_valid_json_number(
    payload: dict[str, Any],
    keys: list[str],
    validator: Any,
) -> float | None:
    """Extract first numeric JSON field that satisfies validator."""
    for key in keys:
        value = payload.get(key)
        with contextlib.suppress(TypeError, ValueError):
            candidate = float(value)
            if validator(candidate):
                return candidate
    return None


def _is_valid_consensus_proportion(value: float) -> bool:
    """Return whether a parsed consensus proportion is finite and in range."""
    return math.isfinite(value) and 0 <= value <= 1


def _is_valid_entropy(value: float) -> bool:
    """Return whether a parsed entropy value is finite and non-negative."""
    return math.isfinite(value) and value >= 0


def _metrics_are_plausible_for_vote_count(
    consensus_proportion: float,
    entropy: float,
    vote_count: int,
) -> bool:
    """Validate that LLM metrics can arise from a finite set of model votes."""
    if vote_count < 1:
        return False
    if not _is_valid_consensus_proportion(consensus_proportion) or not _is_valid_entropy(entropy):
        return False

    supporting_votes = consensus_proportion * vote_count
    nearest_vote_count = round(supporting_votes)
    proportion_is_discrete = (
        1 <= nearest_vote_count <= vote_count and abs(supporting_votes - nearest_vote_count) <= 0.02
    )
    if not proportion_is_discrete:
        return False

    group_sizes = [nearest_vote_count]
    remaining_votes = vote_count - nearest_vote_count
    while remaining_votes > 0:
        next_group_size = min(nearest_vote_count, remaining_votes)
        group_sizes.append(next_group_size)
        remaining_votes -= next_group_size

    minimum_entropy = -sum(
        (group_size / vote_count) * math.log2(group_size / vote_count) for group_size in group_sizes
    )
    majority_share = nearest_vote_count / vote_count
    maximum_entropy = -(majority_share * math.log2(majority_share))
    if nearest_vote_count < vote_count:
        singleton_share = 1 / vote_count
        maximum_entropy -= (
            (vote_count - nearest_vote_count) * singleton_share * math.log2(singleton_share)
        )

    return minimum_entropy - 0.02 <= entropy <= maximum_entropy + 0.02


def _extract_first_valid_json_label(payload: dict[str, Any], keys: list[str]) -> str | None:
    """Extract first non-empty label field from JSON payload."""
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if not isinstance(value, str):
            continue
        normalized = _normalize_predicted_label(value)
        if normalized:
            return normalized
    return None


def _parse_metrics_from_json_text(text: str) -> tuple[float | None, float | None, str | None]:
    """Parse CP/H/label from JSON-style payload when present."""
    json_text = _extract_json_block_from_text(text)
    if not json_text:
        return None, None, None

    try:
        payload = json.loads(json_text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None, None, None

    if not isinstance(payload, dict):
        return None, None, None

    cp_candidates = ["consensus_proportion", "cp", "consensusProportion"]
    entropy_candidates = ["entropy", "h", "shannon_entropy", "shannonEntropy"]
    label_candidates = [
        "majority_prediction",
        "majorityPrediction",
        "majority",
        "annotation",
        "cell_type",
        "label",
    ]

    cp_value = _extract_first_valid_json_number(
        payload,
        cp_candidates,
        _is_valid_consensus_proportion,
    )
    entropy_value = _extract_first_valid_json_number(
        payload,
        entropy_candidates,
        _is_valid_entropy,
    )
    label = _extract_first_valid_json_label(payload, label_candidates)

    if cp_value is not None or entropy_value is not None or label is not None:
        write_log("Parsed consensus metrics from JSON payload", level="debug")
    return cp_value, entropy_value, label


def _extract_numeric_metric(
    *,
    lines: list[str],
    full_text: str,
    line_keywords: list[str],
    inline_patterns: list[str],
    full_patterns: list[str],
    validator: Any,
) -> float | None:
    """Extract numeric metric from either keyworded lines or full-text regex patterns."""
    for line in lines:
        lower_line = line.lower()
        if not any(keyword in lower_line for keyword in line_keywords):
            continue
        for pattern in inline_patterns:
            match = re.search(pattern, line)
            if not match:
                continue
            with contextlib.suppress(ValueError, TypeError):
                value = float(match.group(1))
                if validator(value):
                    return value

    for pattern in full_patterns:
        match = re.search(pattern, full_text)
        if not match:
            continue
        with contextlib.suppress(ValueError, TypeError):
            value = float(match.group(1))
            if validator(value):
                return value
    return None


def _extract_structured_annotation_from_lines(lines: list[str]) -> str | None:
    """Extract annotation from explicitly labeled lines."""
    structured_patterns = [
        r"(?i)majority(?:\s+prediction|\s+cell\s*type)?\s*(?:[:=]|\uFF1A)\s*(.+?)\s*$",
        r"(?i)cell\s*type\s*(?:[:=]|\uFF1A)\s*(.+?)\s*$",
        r"(?i)claim\s*(?:[:=]|\uFF1A)\s*(.+?)\s*$",
    ]
    for line in lines:
        for pattern in structured_patterns:
            match = re.search(pattern, line)
            if not match:
                continue
            normalized = _normalize_predicted_label(match.group(1))
            if normalized:
                return normalized
    return None


def _is_metric_or_meta_line(line_clean: str) -> bool:
    """Return True if a line looks like metric/meta content instead of annotation."""
    metric_patterns = [
        r"^\s*\d+(\.\d+)?\s*$",
        r"^\s*[01]\s*$",
        r"^\s*(0\.\d+|1\.0*|[01])\s*$",
        r"^\s*(\d+\.\d+|\d+)\s*$",
    ]
    label_patterns = ["consensus", "proportion", "entropy", "cp", "majority", "cell type", "claim"]

    if not line_clean:
        return True
    matches_metric_pattern = any(re.match(pattern, line_clean) for pattern in metric_patterns)
    has_metric_prefix = bool(re.match(r"(?i)^\s*(?:cp|h)\s*[:=]", line_clean))
    has_meta_label = any(label in line_clean.lower() for label in label_patterns)
    return matches_metric_pattern or has_metric_prefix or has_meta_label


def _extract_annotation_from_lines(lines: list[str]) -> str | None:
    """Extract annotation from structured lines or unlabeled majority line."""
    structured = _extract_structured_annotation_from_lines(lines)
    if structured:
        return structured

    for line in lines:
        line_clean = line.strip()
        if _is_metric_or_meta_line(line_clean):
            continue
        normalized = _normalize_predicted_label(line_clean)
        if normalized:
            return normalized
    return None


def _extract_metrics_from_text(
    text: str,
) -> tuple[float | None, float | None, str | None]:
    """Extract consensus metrics (CP, H) and optional annotation from text."""
    if not text or not text.strip():
        return None, None, None

    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return None, None, None

    cp_value, entropy_value, annotation = _parse_standard_metrics_lines(lines)
    if cp_value is not None and entropy_value is not None:
        return cp_value, entropy_value, annotation

    write_log(f"Using flexible format parsing for {len(lines)} line(s)", level="debug")

    json_cp, json_entropy, json_annotation = _parse_metrics_from_json_text(text)
    cp_value = json_cp
    entropy_value = json_entropy
    annotation = json_annotation

    if cp_value is None:
        cp_value = _extract_numeric_metric(
            lines=lines,
            full_text=text,
            line_keywords=["consensus", "proportion", "cp"],
            inline_patterns=[
                r"(?i)consensus\s+proportion\s*(?:\(CP\))?\s*[:=]\s*([0-9.]+)",
                r"(?i)\bcp\s*[:=]\s*([0-9.]+)",
            ],
            full_patterns=[
                r"(?i)consensus\s+proportion\s*(?:\(CP\))?\s*[:=]\s*([0-9.]+)",
                r"(?i)\bCP\s*[:=]\s*([0-9.]+)",
            ],
            validator=_is_valid_consensus_proportion,
        )

    if entropy_value is None:
        entropy_value = _extract_numeric_metric(
            lines=lines,
            full_text=text,
            line_keywords=["entropy", "shannon", " h "],
            inline_patterns=[
                r"(?i)(?:shannon\s+)?entropy\s*(?:\(H\))?\s*[:=]\s*([0-9.]+)",
                r"(?i)\bH\s*[:=]\s*([0-9.]+)",
            ],
            full_patterns=[
                r"(?i)(?:shannon\s+)?entropy\s*(?:\(H\))?\s*[:=]\s*([0-9.]+)",
                r"(?i)\bH\s*[:=]\s*([0-9.]+)",
            ],
            validator=_is_valid_entropy,
        )

    if annotation is None:
        annotation = _extract_annotation_from_lines(lines)

    return cp_value, entropy_value, annotation


def _collect_cluster_annotations(
    predictions: dict[str, dict[str, str]],
    cluster: str,
) -> list[str]:
    """Collect normalized non-empty annotations for a single cluster."""
    cluster_annotations: list[str] = []
    for results in predictions.values():
        raw_annotation = results.get(cluster)
        if raw_annotation is None:
            continue
        annotation = str(raw_annotation).strip()
        if annotation and not is_unknown_annotation(annotation):
            cluster_annotations.append(annotation)
    return cluster_annotations


def _collect_candidate_clusters_from_raw_predictions(predictions: dict[str, Any]) -> list[str]:
    """Collect all cluster ids present in raw prediction payloads."""
    cluster_ids: set[str] = set()
    for raw_results in predictions.values():
        if not isinstance(raw_results, dict):
            continue
        for raw_cluster in raw_results:
            if is_missing_value(raw_cluster):
                continue
            cluster_id = str(raw_cluster).strip()
            if cluster_id:
                cluster_ids.add(cluster_id)
    return sorted(cluster_ids, key=cluster_sort_key)


def _annotation_comparison_key(annotation: str) -> str:
    """Normalize conservative nomenclature variants without erasing qualifiers."""
    normalized = annotation.strip().casefold()
    normalized = re.sub(r"[\W_]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"cells$", "cell", normalized)
    normalized = re.sub(r"cytes$", "cyte", normalized)
    normalized = normalized.replace("t lymphocyte", "t cell")
    normalized = normalized.replace("b lymphocyte", "b cell")
    normalized = normalized.replace("natural killer", "nk")
    return normalized


def _resolve_simple_cluster_consensus(
    cluster_annotations: list[str],
    consensus_threshold: float,
    entropy_threshold: float,
) -> tuple[str, float, float] | None:
    """Resolve deterministic consensus cases that already satisfy the contract."""
    if len(cluster_annotations) < 2:
        if cluster_annotations:
            return (
                cluster_annotations[0],
                DEFAULT_FALLBACK_CONSENSUS_PROPORTION,
                DEFAULT_FALLBACK_ENTROPY,
            )
        return "Unknown", 0.0, 0.0

    majority, cp_value, entropy_value = _deterministic_consensus_metrics(cluster_annotations)
    if (
        _has_unique_majority(cluster_annotations)
        and cp_value >= consensus_threshold
        and entropy_value <= entropy_threshold
    ):
        return majority, cp_value, entropy_value

    return None


def _resolve_llm_cluster_consensus(
    *,
    cluster: str,
    cluster_annotations: list[str],
    has_any_api_key: bool,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    api_keys: dict[str, str],
    base_urls: str | dict[str, str] | None,
    consensus_threshold: float,
    entropy_threshold: float,
) -> tuple[str, float, float, bool]:
    """Resolve non-trivial cluster consensus using LLM check and fallback policy."""
    if not has_any_api_key:
        majority, cp_value, entropy_value = _deterministic_consensus_metrics(cluster_annotations)
        write_log(
            f"No API key available for cluster {cluster}; "
            f"using deterministic majority fallback: "
            f"majority={majority}, CP={cp_value:.2f}, H={entropy_value:.2f}",
            level="warning",
        )
        return majority, cp_value, entropy_value, _has_unique_majority(cluster_annotations)

    prompt = create_consensus_check_prompt(
        cluster_annotations,
        consensus_threshold=consensus_threshold,
        entropy_threshold=entropy_threshold,
    )
    llm_response = _call_llm_with_fallback(
        prompt=prompt,
        provider=provider,
        model=model,
        api_key=api_key,
        api_keys=api_keys,
        base_urls=base_urls,
    )

    if llm_response:
        consensus_indicator = _extract_consensus_indicator(llm_response)
        llm_cp, llm_entropy, llm_prediction = _extract_metrics_from_text(llm_response)
        if (
            llm_cp is not None
            and llm_entropy is not None
            and _metrics_are_plausible_for_vote_count(
                llm_cp,
                llm_entropy,
                len(cluster_annotations),
            )
        ):
            resolved_prediction = _normalize_predicted_label(llm_prediction)
            if resolved_prediction is None:
                resolved_prediction, _cp_unused, _entropy_unused = _deterministic_consensus_metrics(
                    cluster_annotations
                )
                write_log(
                    f"LLM metrics parsed but label missing for cluster {cluster}; "
                    f"falling back to deterministic majority '{resolved_prediction}'",
                    level="warning",
                )
            write_log(
                f"LLM consensus check for cluster {cluster}: CP={llm_cp:.2f}, H={llm_entropy:.2f}",
                level="info",
            )
            reached = consensus_indicator is True and resolved_prediction is not None
            return resolved_prediction, llm_cp, llm_entropy, reached
        if llm_cp is not None and llm_entropy is not None:
            write_log(
                f"Ignoring impossible LLM metrics for cluster {cluster}: "
                f"CP={llm_cp}, H={llm_entropy}, votes={len(cluster_annotations)}",
                level="warning",
            )

    write_log(
        f"LLM consensus check failed for cluster {cluster}, returning Unknown",
        level="warning",
    )
    majority, cp_value, entropy_value = _deterministic_consensus_metrics(cluster_annotations)
    write_log(
        f"Using deterministic fallback for cluster {cluster}: "
        f"majority={majority}, CP={cp_value:.2f}, H={entropy_value:.2f}",
        level="warning",
    )
    return majority, cp_value, entropy_value, _has_unique_majority(cluster_annotations)


def _deterministic_consensus_metrics(labels: list[str]) -> tuple[str, float, float]:
    """Compute deterministic metrics from conservative biological label groups."""
    if not labels:
        return "Unknown", DEFAULT_FALLBACK_CONSENSUS_PROPORTION, DEFAULT_FALLBACK_ENTROPY

    counts: dict[str, int] = {}
    representatives: dict[str, str] = {}
    ordered_groups: list[str] = []
    for label in labels:
        group = _annotation_comparison_key(label)
        if group not in counts:
            ordered_groups.append(group)
            representatives[group] = label
            counts[group] = 0
        counts[group] += 1

    majority_group = ordered_groups[0]
    majority_count = counts[majority_group]
    for group in ordered_groups[1:]:
        count = counts[group]
        if count > majority_count:
            majority_group = group
            majority_count = count

    n = len(labels)
    cp_value = majority_count / n

    entropy_value = 0.0
    for count in counts.values():
        p = count / n
        entropy_value -= p * math.log2(p)

    return representatives[majority_group], cp_value, entropy_value


def _has_unique_majority(labels: list[str]) -> bool:
    """Return whether at least two labels produce exactly one largest vote group."""
    if len(labels) < 2:
        return False
    counts: dict[str, int] = {}
    for label in labels:
        group = _annotation_comparison_key(label)
        counts[group] = counts.get(group, 0) + 1
    max_count = max(counts.values())
    return sum(count == max_count for count in counts.values()) == 1


def _extract_structured_cell_type_label(response_text: str) -> str | None:
    """Extract cell-type label from structured discussion response text."""
    patterns = [
        r"(?im)^\s*cell\s*type\s*(?:[:\-]|\uFF1A)\s*(.+?)\s*$",
        r"(?im)^\s*claim\s*(?:[:\-]|\uFF1A)\s*(.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if not match:
            continue
        label = _normalize_predicted_label(match.group(1))
        if label:
            return label
    return None


FREE_TEXT_UNCERTAINTY_PREFIX = re.compile(
    r"(?i)^(?:likely|probably|possibly|maybe|it\s+is|it's|seems\s+to\s+be|appears\s+to\s+be)\s+"
)
FREE_TEXT_NARRATIVE_TOKENS = {
    "candidate",
    "likely",
    "probably",
    "possibly",
    "maybe",
    "uncertain",
    "insufficient",
    "evidence",
    "suggest",
    "suggests",
    "given",
    "without",
    "label",
    "free",
    "text",
    "could",
    "might",
}
KNOWN_FREE_TEXT_CELL_TYPE_ENDING = re.compile(
    r"\b("
    r"lymphocyte|lymphocytes|monocyte|monocytes|macrophage|macrophages|"
    r"neutrophil|neutrophils|fibroblast|fibroblasts|astrocyte|astrocytes|"
    r"neuron|neurons|plasmablast|plasmablasts|megakaryocyte|megakaryocytes|"
    r"erythrocyte|erythrocytes|endothelial|epithelial|myeloid|treg|tregs|nk"
    r")\b$"
)


def _prepare_single_line_label_candidate(response_text: str) -> str | None:
    """Reduce a single-line free-text response to its possible label text."""
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if len(lines) != 1:
        return None

    candidate = re.sub(r"^\s*(?:[-*]\s*)", "", lines[0]).strip()
    candidate = FREE_TEXT_UNCERTAINTY_PREFIX.sub("", candidate).strip()
    return candidate or None


def _is_conservative_cell_type_candidate(candidate: str) -> bool:
    """Return whether a short candidate looks like a cell-type label, not prose."""
    if any(punctuation in candidate for punctuation in ".!?;"):
        return False
    if len(candidate) > 80 or len(candidate.split()) > 10:
        return False

    tokens = {token for token in re.split(r"[^A-Za-z0-9+]+", candidate.lower()) if token}
    if tokens & FREE_TEXT_NARRATIVE_TOKENS:
        return False

    lower_candidate = candidate.lower()
    return bool(
        re.search(r"\bcell(?:s)?\b", lower_candidate)
        or KNOWN_FREE_TEXT_CELL_TYPE_ENDING.search(lower_candidate)
    )


def _extract_conservative_free_text_label(response_text: str) -> str | None:
    """Extract a label from simple free-text responses using conservative heuristics."""
    candidate = _prepare_single_line_label_candidate(response_text)
    if candidate is None or not _is_conservative_cell_type_candidate(candidate):
        return None
    return _normalize_predicted_label(candidate)


def _extract_conservative_multiline_label(response_text: str) -> str | None:
    """Extract a conservative label candidate from multiline free-text responses."""
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return None

    for line in lines:
        candidate = re.sub(r"^\s*(?:[-*]\s*)", "", line).strip()
        if not candidate:
            continue
        if ":" in candidate:
            # Structured key-value lines are handled by structured extraction.
            continue
        label = _extract_conservative_free_text_label(candidate)
        if label:
            return label
    return None


def _fallback_discussion_consensus_from_responses(
    *,
    valid_round_responses: dict[str, str],
    consensus_threshold: float,
    entropy_threshold: float,
) -> dict[str, Any]:
    """Fallback consensus for discussion round when LLM parsing/call fails."""
    extracted_labels = []
    for response in valid_round_responses.values():
        label = _extract_structured_cell_type_label(response)
        if label is None:
            label = _extract_conservative_free_text_label(response)
        if label is None:
            label = _extract_conservative_multiline_label(response)
        if label:
            extracted_labels.append(label)

    if not extracted_labels:
        return DEFAULT_CONSENSUS_RESULT.copy()

    if len(extracted_labels) == 1:
        return {
            "reached": False,
            "consensus_proportion": DEFAULT_FALLBACK_CONSENSUS_PROPORTION,
            "entropy": DEFAULT_FALLBACK_ENTROPY,
            "majority_prediction": extracted_labels[0],
        }

    majority, cp_value, entropy_value = _deterministic_consensus_metrics(extracted_labels)
    reached = (
        _has_unique_majority(extracted_labels)
        and cp_value >= consensus_threshold
        and entropy_value <= entropy_threshold
    )
    return {
        "reached": reached,
        "consensus_proportion": cp_value,
        "entropy": entropy_value,
        "majority_prediction": majority,
    }


def check_consensus(
    predictions: dict[str, Any],
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    api_keys: dict[str, str] | None = None,
    consensus_model: str | dict[str, str] | None = None,
    base_urls: str | dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, float], dict[str, float], list[str]]:
    """Check consensus among different model predictions using LLM assistance.

    Args:
        predictions: Dictionary mapping model names to dictionaries of cluster annotations
        consensus_threshold: Agreement threshold below which a cluster is considered controversial
        entropy_threshold: Entropy threshold above which a cluster is considered controversial
        api_keys: Dictionary mapping provider names to API keys
        consensus_model: Optional model specification for consensus checking.
            Can be a string model name or dict with 'provider'/'model' keys.
            If not provided, picks from api_keys.
        base_urls: Custom base URLs for API endpoints. Can be:
                  - str: Single URL applied to all providers
                  - dict: Provider-specific URLs

    Returns:
        Tuple of (consensus, consensus_proportion, entropy, controversial_clusters)
    """
    consensus = {}
    consensus_proportion = {}
    entropy = {}
    consensus_reached: dict[str, bool] = {}

    consensus_threshold = _normalize_probability(consensus_threshold, "consensus_threshold")
    entropy_threshold = _normalize_nonnegative_number(entropy_threshold, "entropy_threshold")

    if not isinstance(predictions, dict):
        raise ValueError(
            f"predictions must be a dict mapping model->cluster annotations, got {type(predictions).__name__}"
        )

    candidate_clusters = _collect_candidate_clusters_from_raw_predictions(predictions)
    api_keys = _normalize_api_keys(api_keys)
    consensus_model_dict = _normalize_consensus_model_spec(consensus_model)
    predictions = _normalize_predictions(predictions)

    # Filter out models with empty results, but keep models with valid annotations.
    # This allows consensus calculation even if some models failed.
    valid_predictions = {model: results for model, results in predictions.items() if results}

    # Check if we have enough valid predictions
    if not valid_predictions:
        write_log("No valid predictions from any model", level="warning")
        if not candidate_clusters:
            return {}, {}, {}, []
        consensus = {cluster: "Unknown" for cluster in candidate_clusters}
        consensus_proportion = {cluster: 0.0 for cluster in candidate_clusters}
        entropy = {cluster: 0.0 for cluster in candidate_clusters}
        return consensus, consensus_proportion, entropy, candidate_clusters.copy()

    if len(valid_predictions) < len(predictions):
        failed_models = set(predictions.keys()) - set(valid_predictions.keys())
        write_log(
            f"Some models returned empty results, excluding: {failed_models}",
            level="warning",
        )

    # Use filtered predictions for the rest of the function
    predictions = valid_predictions

    # Get all clusters (sorted for deterministic output order)
    # Include raw candidate clusters so blank/None-only clusters are not dropped.
    all_clusters_set: set[str] = set(candidate_clusters)
    for model_results in predictions.values():
        all_clusters_set.update(model_results.keys())
    all_clusters = sorted(all_clusters_set, key=cluster_sort_key)

    primary_provider, primary_model, primary_api_key = _resolve_consensus_provider(
        consensus_model_dict, api_keys
    )
    has_any_api_key = bool(primary_api_key) or _has_supported_api_key(api_keys)

    # Process each cluster
    for cluster in all_clusters:
        if (run := current_execution()) is not None:
            run.cluster = cluster
            run.emit("consensus_cluster")
        cluster_annotations = _collect_cluster_annotations(predictions, cluster)

        simple_result = _resolve_simple_cluster_consensus(
            cluster_annotations,
            consensus_threshold,
            entropy_threshold,
        )
        if simple_result is not None:
            label, cp_value, entropy_value = simple_result
            consensus[cluster] = label
            consensus_proportion[cluster] = cp_value
            entropy[cluster] = entropy_value
            consensus_reached[cluster] = (
                len(cluster_annotations) >= 2
                and _has_unique_majority(cluster_annotations)
                and cp_value >= consensus_threshold
                and entropy_value <= entropy_threshold
            )
            continue

        label, cp_value, entropy_value, semantic_consensus = _resolve_llm_cluster_consensus(
            cluster=cluster,
            cluster_annotations=cluster_annotations,
            has_any_api_key=has_any_api_key,
            provider=primary_provider,
            model=primary_model,
            api_key=primary_api_key,
            api_keys=api_keys,
            base_urls=base_urls,
            consensus_threshold=consensus_threshold,
            entropy_threshold=entropy_threshold,
        )
        consensus[cluster] = label
        consensus_proportion[cluster] = cp_value
        entropy[cluster] = entropy_value
        consensus_reached[cluster] = (
            semantic_consensus
            and cp_value >= consensus_threshold
            and entropy_value <= entropy_threshold
        )

    # Find controversial clusters based on both consensus proportion and entropy
    # (sorted for deterministic output order)
    controversial = sorted(
        (
            cluster
            for cluster, score in consensus_proportion.items()
            if not consensus_reached.get(cluster, False)
            or score < consensus_threshold
            or entropy.get(cluster, 0) > entropy_threshold
        ),
        key=cluster_sort_key,
    )
    return consensus, consensus_proportion, entropy, controversial


def _extract_cell_type_via_llm(
    text: str,
    provider: str,
    model: str,
    api_key: str | None,
    api_keys: dict[str, str] | None = None,
    base_urls: str | dict[str, str] | None = None,
) -> str | None:
    """Use LLM to extract a concise cell-type label from free-text.

    Used for single-response extraction where the discussion consensus
    prompt is not applicable.  The prompt explicitly asks for only the
    cell-type name, and the result is validated with a length bound as
    a sanity check (legitimate cell-type labels are ≤60 characters).

    Args:
        text: Free-text response containing a cell-type reference
        provider: LLM provider to use
        model: LLM model to use
        api_key: API key for the provider
        api_keys: Dictionary of API keys (for fallback)
        base_urls: Custom base URLs for API endpoints

    Returns:
        str | None: Clean cell-type label or None if extraction failed
    """
    prompt = create_cell_type_extraction_prompt(text)
    response = _call_llm_with_fallback(
        prompt=prompt,
        provider=provider,
        model=model,
        api_key=api_key,
        api_keys=api_keys,
        base_urls=base_urls,
    )
    if response:
        cell_type = _normalize_predicted_label(response)
        if cell_type and len(cell_type) <= 60:
            write_log(f"LLM extracted cell type: '{cell_type}'", level="debug")
            return cell_type
        write_log(
            f"LLM extraction returned unsuitable result: '{response.strip()}'",
            level="warning",
        )
    return None


def check_consensus_for_discussion_round(
    round_responses: dict[str, Any],
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    api_keys: dict[str, str] | None = None,
    consensus_model: str | dict[str, str] | None = None,
    base_urls: str | dict[str, str] | None = None,
) -> dict[str, Any]:
    """Check consensus among model responses for a single discussion round.

    LLM-native approach mirroring R's check_consensus (check_consensus.R:488):
    the LLM receives raw discussion responses and performs both cell-type
    extraction and consensus checking in a single call, avoiding the
    sentence-as-label problem that arises from regex-based extraction.

    1. Empty input: returns default Unknown result immediately (no LLM needed)
    2. Single response: LLM extraction to produce a clean label
    3. Multiple responses: ONE LLM call with raw responses

    Args:
        round_responses: Dictionary mapping model names to their responses
        consensus_threshold: Agreement threshold (default: 0.7)
        entropy_threshold: Entropy threshold (default: 1.0)
        api_keys: Dictionary mapping provider names to API keys
        consensus_model: Optional model specification for consensus checking.
            Can be a string model name or dict with 'provider'/'model' keys.
        base_urls: Custom base URLs for API endpoints

    Returns:
        dict containing:
            - reached: bool, whether consensus was reached
            - consensus_proportion: float, the consensus proportion
            - entropy: float, the Shannon entropy
            - majority_prediction: str, the majority cell type prediction
    """
    consensus_threshold = _normalize_probability(consensus_threshold, "consensus_threshold")
    entropy_threshold = _normalize_nonnegative_number(entropy_threshold, "entropy_threshold")
    valid_round_responses = _collect_valid_round_responses(round_responses)

    # Empty — nothing to check, no LLM needed
    if not valid_round_responses:
        write_log("No responses to check consensus", level="warning")
        return DEFAULT_CONSENSUS_RESULT.copy()

    api_keys = _normalize_api_keys(api_keys)
    consensus_model_dict = _normalize_consensus_model_spec(consensus_model)

    primary_provider, primary_model, primary_api_key = _resolve_consensus_provider(
        consensus_model_dict, api_keys
    )

    # Single response — first try deterministic structured extraction,
    # then optionally use LLM extraction as a fallback enhancer.
    if len(valid_round_responses) == 1:
        deterministic_result = _fallback_discussion_consensus_from_responses(
            valid_round_responses=valid_round_responses,
            consensus_threshold=consensus_threshold,
            entropy_threshold=entropy_threshold,
        )
        deterministic_prediction = _normalize_predicted_label(
            deterministic_result.get("majority_prediction")
        )
        if deterministic_prediction:
            write_log(
                "Only 1 response, using structured label extraction without LLM",
                level="info",
            )
            deterministic_result["majority_prediction"] = deterministic_prediction
            return deterministic_result

        write_log("Only 1 response, extracting label via LLM", level="warning")
        single_response = next(iter(valid_round_responses.values()))
        cell_type = _extract_cell_type_via_llm(
            single_response,
            primary_provider,
            primary_model,
            primary_api_key,
            api_keys=api_keys,
            base_urls=base_urls,
        )
        return {
            "reached": False,
            "consensus_proportion": DEFAULT_FALLBACK_CONSENSUS_PROPORTION,
            "entropy": DEFAULT_FALLBACK_ENTROPY,
            "majority_prediction": cell_type or "Unknown",
        }

    # --- Primary path: LLM-native consensus check ---
    # Send raw responses to LLM for simultaneous extraction + consensus.
    # This mirrors R's approach (check_consensus.R:488) where the LLM
    # receives full discussion responses, extracts clean cell-type labels,
    # and calculates consensus metrics — all in one call.
    prompt = create_discussion_consensus_prompt(
        valid_round_responses,
        consensus_threshold=consensus_threshold,
        entropy_threshold=entropy_threshold,
    )

    llm_response = _call_llm_with_fallback(
        prompt=prompt,
        provider=primary_provider,
        model=primary_model,
        api_key=primary_api_key,
        api_keys=api_keys,
        base_urls=base_urls,
    )

    if llm_response:
        consensus_indicator = _extract_consensus_indicator(llm_response)
        cp_value, entropy_value, llm_prediction = _extract_metrics_from_text(llm_response)

        if (
            cp_value is not None
            and entropy_value is not None
            and _metrics_are_plausible_for_vote_count(
                cp_value,
                entropy_value,
                len(valid_round_responses),
            )
        ):
            resolved_prediction = _normalize_predicted_label(llm_prediction)
            if resolved_prediction is None:
                fallback_result = _fallback_discussion_consensus_from_responses(
                    valid_round_responses=valid_round_responses,
                    consensus_threshold=consensus_threshold,
                    entropy_threshold=entropy_threshold,
                )
                fallback_prediction = _normalize_predicted_label(
                    fallback_result.get("majority_prediction")
                )
                if fallback_prediction:
                    resolved_prediction = fallback_prediction
                    write_log(
                        "LLM metrics parsed but label missing; using structured-response label fallback",
                        level="warning",
                    )

            reached = (
                consensus_indicator is True
                and resolved_prediction is not None
                and cp_value >= consensus_threshold
                and entropy_value <= entropy_threshold
            )

            write_log(
                f"LLM consensus check: CP={cp_value:.2f}, H={entropy_value:.2f}, "
                f"reached={reached}, prediction={resolved_prediction}",
                level="info",
            )

            return {
                "reached": reached,
                "consensus_proportion": cp_value,
                "entropy": entropy_value,
                "majority_prediction": resolved_prediction or "Unknown",
            }
        if cp_value is not None and entropy_value is not None:
            write_log(
                "Ignoring impossible discussion consensus metrics: "
                f"CP={cp_value}, H={entropy_value}, votes={len(valid_round_responses)}",
                level="warning",
            )

    # LLM call failed/returned unparseable output.
    # Fall back to structured text extraction from discussion responses.
    fallback_result = _fallback_discussion_consensus_from_responses(
        valid_round_responses=valid_round_responses,
        consensus_threshold=consensus_threshold,
        entropy_threshold=entropy_threshold,
    )
    write_log(
        "LLM consensus check failed; using structured-response fallback",
        level="warning",
    )
    return fallback_result


def _build_discussion_model_info(
    models: list[_ResolvedModelSpec],
    api_keys: dict[str, str],
    base_urls: str | dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Build runnable discussion model descriptors with de-dup and key resolution."""
    model_info_list: list[dict[str, Any]] = []
    seen_model_keys: set[str] = set()

    for model_spec in models:
        api_key = api_keys.get(model_spec.provider) or load_api_key(model_spec.provider)
        if not api_key:
            write_log(
                f"No API key for {model_spec.provider}, skipping {model_spec.model}",
                level="warning",
            )
            continue

        if model_spec.key in seen_model_keys:
            write_log(
                f"Duplicate discussion model '{model_spec.key}' detected, skipping",
                level="warning",
            )
            continue
        seen_model_keys.add(model_spec.key)

        model_info_list.append(
            {
                "key": model_spec.key,
                "name": model_spec.model,
                "provider": model_spec.provider,
                "api_key": api_key,
                "base_url": resolve_provider_base_url(model_spec.provider, base_urls),
            }
        )

    return model_info_list


def _build_discussion_round_prompt(
    *,
    cluster_id: str,
    marker_genes: list[str],
    rounds_history: list[dict[str, str]],
    initial_predictions: dict[str, str],
    current_round: int,
    species: str,
    tissue: str | None,
) -> str:
    """Build prompt for a single discussion round."""
    if current_round == 1:
        return create_initial_discussion_prompt(
            cluster_id=cluster_id,
            marker_genes=marker_genes,
            initial_predictions=initial_predictions,
            species=species,
            tissue=tissue,
        )

    return create_discussion_prompt(
        cluster_id=cluster_id,
        marker_genes=marker_genes,
        previous_rounds=rounds_history,
        round_number=current_round,
        species=species,
        tissue=tissue,
    )


def _run_discussion_round(
    *,
    model_info_list: list[dict[str, Any]],
    prompt: str,
    current_round: int,
    use_cache: bool,
    force_rerun: bool,
    cache_dir: str | None,
) -> dict[str, str]:
    """Execute one discussion round and collect responses from all models."""
    round_responses: dict[str, str] = {}
    for model_info in model_info_list:
        model_key = model_info["key"]
        try:
            response = get_model_response(
                prompt=prompt,
                provider=model_info["provider"],
                model=model_info["name"],
                api_key=model_info["api_key"],
                use_cache=use_cache and not force_rerun,
                cache_dir=cache_dir,
                base_url=model_info["base_url"],
            )
            round_responses[model_key] = response
            write_log(f"Got response from {model_key} in round {current_round}")
        except RECOVERABLE_LLM_EXCEPTIONS as e:
            write_log(
                f"Error getting response from {model_key}: {e!s}",
                level="warning",
            )
            round_responses[model_key] = f"Error: {e!s}"
    return round_responses


def _resolve_last_round_decision(
    *,
    rounds_history: list[dict[str, str]],
    consensus_threshold: float,
    entropy_threshold: float,
    api_keys: dict[str, str],
    consensus_model: str | dict[str, str] | None,
    base_urls: str | dict[str, str] | None,
) -> tuple[str | None, float, float]:
    """Resolve final decision from last discussion round when no early consensus was reached."""
    if not rounds_history:
        return None, DEFAULT_FALLBACK_CONSENSUS_PROPORTION, DEFAULT_FALLBACK_ENTROPY

    last_round = rounds_history[-1]
    valid_last = _collect_valid_round_responses(last_round)
    if not valid_last:
        return None, DEFAULT_FALLBACK_CONSENSUS_PROPORTION, DEFAULT_FALLBACK_ENTROPY

    last_consensus = check_consensus_for_discussion_round(
        round_responses=valid_last,
        consensus_threshold=consensus_threshold,
        entropy_threshold=entropy_threshold,
        api_keys=api_keys,
        consensus_model=consensus_model,
        base_urls=base_urls,
    )
    decision = last_consensus["majority_prediction"]
    if not isinstance(decision, str) or is_unknown_annotation(decision):
        decision = None

    return (
        decision,
        last_consensus["consensus_proportion"],
        last_consensus["entropy"],
    )


def _build_cluster_initial_predictions(
    model_predictions: dict[str, dict[str, str]],
    cluster_id: str,
) -> dict[str, str]:
    """Build initial per-model annotations for a target cluster."""
    initial_predictions: dict[str, str] = {}
    for model_name, predictions in model_predictions.items():
        if cluster_id not in predictions:
            continue
        annotation = normalize_annotation(predictions.get(cluster_id, "Unknown"))
        if annotation == "Unknown":
            continue
        initial_predictions[model_name] = annotation
    return initial_predictions


def _run_cluster_discussion_rounds(
    *,
    cluster_id: str,
    marker_genes: list[str],
    initial_predictions: dict[str, str],
    model_info_list: list[dict[str, Any]],
    species: str,
    tissue: str | None,
    max_discussion_rounds: int,
    consensus_threshold: float,
    entropy_threshold: float,
    api_keys: dict[str, str],
    consensus_model: str | dict[str, str] | None,
    use_cache: bool,
    force_rerun: bool,
    cache_dir: str | None,
    base_urls: str | dict[str, str] | None,
) -> tuple[str | None, list[dict[str, str]], float, float]:
    """Execute discussion rounds for a cluster and return final decision + metrics."""
    rounds_history: list[dict[str, str]] = []
    final_decision: str | None = None
    current_cp = DEFAULT_FALLBACK_CONSENSUS_PROPORTION
    current_h = DEFAULT_FALLBACK_ENTROPY

    last_consensus = None
    for current_round in range(1, max_discussion_rounds + 1):
        if (run := current_execution()) is not None:
            run.round = current_round
            run.emit("discussion_round")
        write_log(f"Starting round {current_round} for cluster {cluster_id}")

        prompt = _build_discussion_round_prompt(
            cluster_id=cluster_id,
            marker_genes=marker_genes,
            rounds_history=rounds_history,
            initial_predictions=initial_predictions,
            current_round=current_round,
            species=species,
            tissue=tissue,
        )

        round_responses = _run_discussion_round(
            model_info_list=model_info_list,
            prompt=prompt,
            current_round=current_round,
            use_cache=use_cache,
            force_rerun=force_rerun,
            cache_dir=cache_dir,
        )
        rounds_history.append(round_responses)

        valid_responses = _collect_valid_round_responses(round_responses)
        if len(valid_responses) < 2:
            write_log(
                f"Only {len(valid_responses)} valid responses in round {current_round}",
                level="warning",
            )
            if current_execution() is not None:
                break
            continue

        consensus_result = check_consensus_for_discussion_round(
            round_responses=valid_responses,
            consensus_threshold=consensus_threshold,
            entropy_threshold=entropy_threshold,
            api_keys=api_keys,
            consensus_model=consensus_model,
            base_urls=base_urls,
        )

        last_consensus = consensus_result
        current_cp = consensus_result["consensus_proportion"]
        current_h = consensus_result["entropy"]
        majority = consensus_result["majority_prediction"]

        write_log(
            f"Round {current_round} consensus: CP={current_cp:.2f}, H={current_h:.2f}, "
            f"majority={majority}, reached={consensus_result['reached']}",
            level="info",
        )

        if consensus_result["reached"]:
            final_decision = majority
            write_log(
                f"Consensus reached in round {current_round} for cluster {cluster_id}: {final_decision}",
                level="info",
            )
            break

    if not final_decision and last_consensus is not None:
        final_decision = last_consensus["majority_prediction"]
    elif not final_decision and rounds_history and current_execution() is not None:
        fallback = _fallback_discussion_consensus_from_responses(
            valid_round_responses=_collect_valid_round_responses(rounds_history[-1]),
            consensus_threshold=consensus_threshold,
            entropy_threshold=entropy_threshold,
        )
        final_decision = fallback["majority_prediction"]
        current_cp, current_h = fallback["consensus_proportion"], fallback["entropy"]
    elif not final_decision and rounds_history:
        (
            last_round_decision,
            last_round_cp,
            last_round_h,
        ) = _resolve_last_round_decision(
            rounds_history=rounds_history,
            consensus_threshold=consensus_threshold,
            entropy_threshold=entropy_threshold,
            api_keys=api_keys,
            consensus_model=consensus_model,
            base_urls=base_urls,
        )
        final_decision = last_round_decision
        current_cp = last_round_cp
        current_h = last_round_h
        if last_round_decision:
            write_log(
                f"Using last round majority for cluster {cluster_id}: {last_round_decision} "
                f"(CP={current_cp:.2f}, H={current_h:.2f})",
                level="info",
            )

    return final_decision, rounds_history, current_cp, current_h


def _store_cluster_discussion_outcome(
    *,
    cluster_id: str,
    final_decision: str | None,
    rounds_history: list[dict[str, str]],
    current_cp: float,
    current_h: float,
    results: dict[str, str],
    discussion_history: dict[str, list[dict]],
    updated_consensus_proportion: dict[str, float],
    updated_entropy: dict[str, float],
) -> None:
    """Store final cluster result and metrics into aggregate output dictionaries."""
    if final_decision and not is_unknown_annotation(final_decision):
        results[cluster_id] = normalize_annotation(final_decision)
    else:
        results[cluster_id] = "Unknown"

    discussion_history[cluster_id] = rounds_history
    updated_consensus_proportion[cluster_id] = current_cp
    updated_entropy[cluster_id] = current_h


def _normalize_discussion_model_predictions(
    model_predictions: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Normalize discussion model predictions to trimmed model/cluster keys."""
    normalized_predictions: dict[str, dict[str, Any]] = {}
    seen_model_names: set[str] = set()
    for raw_model_name, raw_predictions in model_predictions.items():
        model_name = _normalize_model_identity(
            raw_model_name,
            "discussion prediction model names",
        )
        if model_name in seen_model_names:
            raise ValueError(
                f"Duplicate discussion model name after whitespace normalization: '{model_name}'"
            )
        seen_model_names.add(model_name)
        if not isinstance(raw_predictions, dict):
            write_log(
                f"Model '{model_name}' discussion predictions must be a dict, "
                f"got {type(raw_predictions).__name__}; skipping",
                level="warning",
            )
            continue

        normalized_cluster_map: dict[str, Any] = {}
        for raw_cluster_id, annotation in raw_predictions.items():
            if is_missing_value(raw_cluster_id):
                continue
            if is_missing_value(annotation):
                continue
            cluster_id = str(raw_cluster_id).strip()
            if cluster_id:
                if not isinstance(annotation, str):
                    raise ValueError(
                        f"Model '{model_name}' discussion annotation for cluster "
                        f"'{cluster_id}' must be a string"
                    )
                normalized_annotation = normalize_annotation(annotation)
                if normalized_annotation == "Unknown":
                    continue
                normalized_cluster_map[cluster_id] = normalized_annotation

        normalized_predictions[model_name] = normalized_cluster_map

    return normalized_predictions


def _normalize_controversial_cluster_ids(controversial_clusters: list[Any]) -> list[str]:
    """Normalize controversial cluster IDs with trim/filter/dedup semantics."""
    is_unordered_input = isinstance(controversial_clusters, set)
    normalized_clusters: list[str] = []
    seen: set[str] = set()
    skipped_invalid = 0
    for raw_cluster_id in controversial_clusters:
        if is_missing_value(raw_cluster_id):
            skipped_invalid += 1
            continue
        cluster_id = str(raw_cluster_id).strip()
        if not cluster_id:
            skipped_invalid += 1
            continue
        if cluster_id not in seen:
            seen.add(cluster_id)
            normalized_clusters.append(cluster_id)

    if skipped_invalid:
        write_log(
            f"Ignored {skipped_invalid} invalid controversial cluster IDs after normalization",
            level="warning",
        )

    if is_unordered_input:
        normalized_clusters = sorted(normalized_clusters, key=cluster_sort_key)

    return normalized_clusters


def process_controversial_clusters(
    marker_genes: dict[str, list[str]],
    controversial_clusters: list[str],
    model_predictions: dict[str, dict[str, str]],
    species: str,
    tissue: str | None = None,
    models: list[ModelSpecInput | _ResolvedModelSpec] | None = None,
    api_keys: dict[str, str] | None = None,
    max_discussion_rounds: int = 3,
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    use_cache: bool = True,
    cache_dir: str | None = None,
    base_urls: str | dict[str, str] | None = None,
    force_rerun: bool = False,
    consensus_model: str | dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, list[dict]], dict[str, float], dict[str, float]]:
    """Process controversial clusters through multi-model discussion.

    This function facilitates a real discussion between multiple LLMs, where each
    model can see and respond to other models' arguments. This mirrors the R
    implementation where all models participate in each round of discussion.

    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes
        controversial_clusters: List of controversial cluster IDs
        model_predictions: Dictionary mapping model names to dictionaries of
            cluster annotations (from initial annotation phase)
        species: Species name (e.g., 'human', 'mouse')
        tissue: Optional tissue name (e.g., 'brain', 'liver')
        models: List of models to participate in the discussion. Each item can be
            a string (model name) or dict with 'provider' and 'model' keys.
        api_keys: Dictionary mapping provider names to API keys
        max_discussion_rounds: Maximum number of discussion rounds
        consensus_threshold: Consensus proportion threshold for agreement
        entropy_threshold: Entropy threshold for agreement
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files
        base_urls: Custom base URLs for API endpoints
        force_rerun: If True, ignore cached results
        consensus_model: Optional model specification for consensus checking.
            Can be a string model name or dict with 'provider'/'model' keys.
            If not provided, picks from the caller's api_keys.

    Returns:
        tuple containing:
            - results: Dict mapping cluster IDs to resolved annotations
            - discussion_history: Dict mapping cluster IDs to list of round responses
            - updated_consensus_proportion: Dict mapping cluster IDs to CP scores
            - updated_entropy: Dict mapping cluster IDs to entropy scores
    """
    max_discussion_rounds = _normalize_nonnegative_integer(
        max_discussion_rounds, "max_discussion_rounds"
    )
    consensus_threshold = _normalize_probability(consensus_threshold, "consensus_threshold")
    entropy_threshold = _normalize_nonnegative_number(entropy_threshold, "entropy_threshold")
    use_cache = validate_bool(use_cache, "use_cache")
    force_rerun = validate_bool(force_rerun, "force_rerun")
    species = normalize_text(species, "species", required=True)
    tissue = normalize_text(tissue, "tissue")

    if not models:
        write_log("No models provided for discussion", level="error")
        return {}, {}, {}, {}

    if isinstance(controversial_clusters, (str, bytes)) or not isinstance(
        controversial_clusters, (list, tuple, set)
    ):
        raise ValueError(
            "controversial_clusters must be a list/tuple/set of cluster IDs, not a string"
        )

    if not isinstance(model_predictions, dict):
        raise ValueError(
            "model_predictions must be a dict mapping model name to cluster predictions"
        )

    resolved_models = _normalize_models_spec(models)

    api_keys = _normalize_api_keys(api_keys)
    consensus_model_dict = _normalize_consensus_model_spec(consensus_model)
    api_keys = _ensure_consensus_model_api_key(api_keys, consensus_model_dict)

    # Normalize cluster keys for consistent lookups.
    marker_genes = normalize_marker_genes_keys(marker_genes)
    model_predictions = _normalize_discussion_model_predictions(model_predictions)
    controversial_clusters = _normalize_controversial_cluster_ids(controversial_clusters)

    results = {}
    discussion_history = {}
    updated_consensus_proportion = {}
    updated_entropy = {}

    model_info_list = _build_discussion_model_info(resolved_models, api_keys, base_urls)

    if len(model_info_list) < 2:
        write_log(
            f"Need at least 2 models for discussion, only have {len(model_info_list)}",
            level="error",
        )
        return {}, {}, {}, {}

    write_log(f"Starting multi-model discussion with {len(model_info_list)} models")

    for cluster_id in controversial_clusters:
        if (run := current_execution()) is not None:
            run.phase, run.cluster = "discussion", cluster_id
            model_info_list = [m for m in model_info_list if m["key"] not in run.unavailable]
            if len(model_info_list) < 2:
                run.emit("discussion_stopped", reason="insufficient_available_models")
                break
        write_log(f"Processing controversial cluster {cluster_id}")

        current_marker_genes = marker_genes.get(cluster_id, [])
        if not current_marker_genes:
            write_log(f"No marker genes found for cluster {cluster_id}", level="warning")
            results[cluster_id] = "Unknown"
            discussion_history[cluster_id] = []
            updated_consensus_proportion[cluster_id] = DEFAULT_FALLBACK_CONSENSUS_PROPORTION
            updated_entropy[cluster_id] = DEFAULT_FALLBACK_ENTROPY
            continue

        initial_predictions = _build_cluster_initial_predictions(model_predictions, cluster_id)
        if current_execution() is not None and len(initial_predictions) < 2:
            write_log(f"Skipping discussion for {cluster_id}: insufficient initial evidence")
            continue

        try:
            final_decision, rounds_history, current_cp, current_h = _run_cluster_discussion_rounds(
                cluster_id=cluster_id,
                marker_genes=current_marker_genes,
                initial_predictions=initial_predictions,
                model_info_list=model_info_list,
                species=species,
                tissue=tissue,
                max_discussion_rounds=max_discussion_rounds,
                consensus_threshold=consensus_threshold,
                entropy_threshold=entropy_threshold,
                api_keys=api_keys,
                consensus_model=consensus_model_dict,
                use_cache=use_cache,
                force_rerun=force_rerun,
                cache_dir=cache_dir,
                base_urls=base_urls,
            )
            _store_cluster_discussion_outcome(
                cluster_id=cluster_id,
                final_decision=final_decision,
                rounds_history=rounds_history,
                current_cp=current_cp,
                current_h=current_h,
                results=results,
                discussion_history=discussion_history,
                updated_consensus_proportion=updated_consensus_proportion,
                updated_entropy=updated_entropy,
            )

        except RECOVERABLE_LLM_EXCEPTIONS as e:
            write_log(f"Error during discussion for cluster {cluster_id}: {e!s}", level="error")
            results[cluster_id] = "Unknown"
            discussion_history[cluster_id] = []
            updated_consensus_proportion[cluster_id] = DEFAULT_FALLBACK_CONSENSUS_PROPORTION
            updated_entropy[cluster_id] = DEFAULT_FALLBACK_ENTROPY

    return results, discussion_history, updated_consensus_proportion, updated_entropy


def _filter_marker_genes_for_clusters(
    marker_genes: dict[str, list[str]],
    clusters_to_analyze: list[str] | None,
) -> dict[str, list[str]]:
    """Filter marker genes to requested clusters with validation and stable order."""
    if clusters_to_analyze is None:
        return marker_genes

    if isinstance(clusters_to_analyze, (str, bytes)) or not isinstance(
        clusters_to_analyze, (list, tuple, set)
    ):
        raise ValueError(
            "clusters_to_analyze must be a list/tuple/set of cluster IDs, not a string"
        )

    raw_cluster_iterable = (
        sorted(clusters_to_analyze, key=lambda cluster_id: cluster_sort_key(str(cluster_id)))
        if isinstance(clusters_to_analyze, set)
        else clusters_to_analyze
    )

    requested_clusters: list[str] = []
    skipped_invalid: list[str] = []
    for cluster_id in raw_cluster_iterable:
        if is_missing_value(cluster_id):
            skipped_invalid.append(str(cluster_id))
            continue
        normalized = str(cluster_id).strip()
        if not normalized:
            skipped_invalid.append(str(cluster_id))
            continue
        requested_clusters.append(normalized)

    # Deduplicate while preserving caller-specified order.
    requested_clusters = list(dict.fromkeys(requested_clusters))

    if skipped_invalid:
        write_log(
            "Ignored missing or empty cluster IDs in clusters_to_analyze",
            level="warning",
        )

    available_clusters = list(marker_genes.keys())
    valid_clusters = [
        cluster_id for cluster_id in requested_clusters if cluster_id in available_clusters
    ]
    invalid_clusters = [
        cluster_id for cluster_id in requested_clusters if cluster_id not in available_clusters
    ]

    if invalid_clusters:
        warning_msg = (
            f"The following cluster IDs were not found in the input: {', '.join(invalid_clusters)}"
        )
        write_log(warning_msg, level="warning")

    if not valid_clusters:
        error_msg = "None of the specified clusters exist in the input data."
        write_log(error_msg, level="error")
        raise ValueError(error_msg)

    filtered = {cluster_id: marker_genes[cluster_id] for cluster_id in valid_clusters}
    log_msg = f"Filtered to analyze {len(valid_clusters)} clusters: {', '.join(valid_clusters)}"
    write_log(log_msg)
    return filtered


def _ensure_consensus_model_api_key(
    api_keys: dict[str, str],
    consensus_model_dict: dict[str, str] | None,
) -> dict[str, str]:
    """Ensure consensus-model provider key is available when provider is specified."""
    if not consensus_model_dict:
        return api_keys

    consensus_provider = consensus_model_dict.get("provider")
    if consensus_provider and consensus_provider not in api_keys:
        api_key = load_api_key(consensus_provider)
        if api_key:
            api_keys[consensus_provider] = api_key
    return api_keys


def _run_initial_annotations(
    *,
    marker_genes: dict[str, list[str]],
    species: str,
    models: list[_ResolvedModelSpec],
    api_keys: dict[str, str],
    tissue: str | None,
    additional_context: str | None,
    use_cache: bool,
    force_rerun: bool,
    cache_dir: str | None,
    base_urls: str | dict[str, str] | None,
    verbose: bool,
    prompt_template: str | None = None,
) -> dict[str, dict[str, str]]:
    """Run initial annotation phase across models with de-dup and error isolation."""
    model_results: dict[str, dict[str, str]] = {}
    seen_model_keys: set[str] = set()

    for model_spec in models:
        api_key = api_keys.get(model_spec.provider)
        if not api_key:
            write_log(
                f"Warning: No API key found for {model_spec.provider}, skipping {model_spec.model}",
                level="warning",
            )
            continue

        if model_spec.key in seen_model_keys:
            write_log(
                f"Warning: Duplicate model '{model_spec.key}' detected, skipping",
                level="warning",
            )
            continue
        seen_model_keys.add(model_spec.key)

        if verbose:
            write_log(f"Annotating with {model_spec.key}")

        try:
            results = annotate_clusters(
                marker_genes=marker_genes,
                species=species,
                provider=model_spec.provider,
                model=model_spec.model,
                api_key=api_key,
                tissue=tissue,
                additional_context=additional_context,
                prompt_template=prompt_template,
                use_cache=use_cache and not force_rerun,
                cache_dir=cache_dir,
                base_urls=base_urls,
            )
            model_results[model_spec.key] = results
            if verbose:
                write_log(f"Successfully annotated with {model_spec.key}")
        except RECOVERABLE_LLM_EXCEPTIONS as e:
            write_log(f"Error annotating with {model_spec.key}: {e!s}", level="error")

    return model_results


def _clean_annotations(annotations: dict[str, Any]) -> dict[str, str]:
    """Normalize annotation dictionary to non-empty strings."""
    cleaned: dict[str, str] = {}
    for cid, ann in annotations.items():
        cleaned[cid] = "Unknown" if is_unknown_annotation(ann) else str(ann).strip()
    return cleaned


def _prepare_interactive_annotation_context(
    *,
    marker_genes: dict[str, list[str]],
    species: str,
    tissue: str | None,
    models: list[str | dict[str, str]] | None,
    api_keys: dict[str, str] | None,
    consensus_model: str | dict[str, str] | None,
    clusters_to_analyze: list[str] | None,
    consensus_threshold: float,
    entropy_threshold: float,
    max_discussion_rounds: int,
    verbose: bool,
) -> tuple[
    dict[str, Any],
    dict[str, list[str]],
    list[_ResolvedModelSpec],
    dict[str, str],
    dict[str, str] | None,
]:
    """Prepare normalized context for interactive consensus orchestration."""
    resolved_models = _normalize_models_spec(models)

    metadata = _build_metadata(
        models=models,
        species=species,
        tissue=tissue,
        consensus_threshold=consensus_threshold,
        entropy_threshold=entropy_threshold,
        max_discussion_rounds=max_discussion_rounds,
    )

    marker_genes_normalized = normalize_marker_genes_keys(marker_genes)
    if verbose:
        write_log("Starting interactive consensus annotation")
    marker_genes_filtered = _filter_marker_genes_for_clusters(
        marker_genes_normalized,
        clusters_to_analyze,
    )

    resolved_api_keys = _resolve_api_keys_for_models(resolved_models, api_keys)
    consensus_model_dict = _normalize_consensus_model_spec(consensus_model)
    resolved_api_keys = _ensure_consensus_model_api_key(resolved_api_keys, consensus_model_dict)

    return metadata, marker_genes_filtered, resolved_models, resolved_api_keys, consensus_model_dict


def _update_metrics_for_resolved_clusters(
    consensus_proportion: dict[str, float],
    entropy: dict[str, float],
    updated_cp: dict[str, float],
    updated_h: dict[str, float],
) -> None:
    """Merge updated metrics from discussion resolution into consensus metrics."""
    for cluster_id, cp_value in updated_cp.items():
        consensus_proportion[cluster_id] = cp_value
    for cluster_id, h_value in updated_h.items():
        entropy[cluster_id] = h_value


def _resolve_controversial_clusters_if_needed(
    *,
    controversial: list[str],
    marker_genes: dict[str, list[str]],
    model_results: dict[str, dict[str, str]],
    species: str,
    tissue: str | None,
    models: list[_ResolvedModelSpec],
    api_keys: dict[str, str],
    max_discussion_rounds: int,
    consensus_threshold: float,
    entropy_threshold: float,
    use_cache: bool,
    cache_dir: str | None,
    base_urls: str | dict[str, str] | None,
    force_rerun: bool,
    consensus_model: dict[str, str] | None,
    verbose: bool,
) -> tuple[dict[str, str], dict[str, list[dict]], dict[str, float], dict[str, float]]:
    """Resolve controversial clusters via discussion, returning resolution + updated metrics."""
    if not controversial:
        return {}, {}, {}, {}

    if verbose:
        write_log(
            f"Resolving {len(controversial)} controversial clusters through multi-model discussion"
        )

    try:
        return process_controversial_clusters(
            marker_genes=marker_genes,
            controversial_clusters=controversial,
            model_predictions=model_results,
            species=species,
            tissue=tissue,
            models=models,
            api_keys=api_keys,
            max_discussion_rounds=max_discussion_rounds,
            consensus_threshold=consensus_threshold,
            entropy_threshold=entropy_threshold,
            use_cache=use_cache,
            cache_dir=cache_dir,
            base_urls=base_urls,
            force_rerun=force_rerun,
            consensus_model=consensus_model,
        )
    except RECOVERABLE_LLM_EXCEPTIONS as e:
        write_log(f"Error resolving controversial clusters: {e!s}", level="error")
        return {}, {}, {}, {}


def _merge_consensus_and_resolved(
    consensus: dict[str, str],
    resolved: dict[str, str],
) -> dict[str, str]:
    """Merge resolved controversial results into base consensus annotations.

    Principle:
    - Never downgrade a known base consensus label to Unknown.
    - Still allow Unknown to fill missing clusters when base consensus lacks them.
    """
    final_annotations = consensus.copy()
    for cluster_id, resolved_label in resolved.items():
        normalized_resolved = normalize_annotation(resolved_label)
        base_label = final_annotations.get(cluster_id)

        if (
            normalized_resolved == "Unknown"
            and base_label is not None
            and not is_unknown_annotation(base_label)
        ):
            # Keep known base consensus when discussion adds no better signal.
            continue

        final_annotations[cluster_id] = normalized_resolved
    return final_annotations


def interactive_consensus_annotation(
    marker_genes: dict[str, list[str]],
    species: str,
    models: list[str | dict[str, str]] | None = None,
    api_keys: dict[str, str] | None = None,
    tissue: str | None = None,
    additional_context: str | None = None,
    consensus_threshold: float = 0.7,
    entropy_threshold: float = 1.0,
    max_discussion_rounds: int = 3,
    use_cache: bool = True,
    cache_dir: str | None = None,
    verbose: bool = False,
    consensus_model: str | dict[str, str] | None = None,
    base_urls: str | dict[str, str] | None = None,
    clusters_to_analyze: list[str] | None = None,
    force_rerun: bool = False,
    prompt_template: str | None = None,
    *,
    request_timeout: float = 120,
    max_runtime: float = 900,
    progress_callback=None,
    should_cancel=None,
) -> dict[str, Any]:
    """Perform consensus annotation of cell types using multiple LLMs and interactive resolution.

    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes
        species: Species name (e.g., 'human', 'mouse')
        models: List of models to use for annotation
        api_keys: Dictionary mapping provider names to API keys
        tissue: Optional tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        consensus_threshold: Agreement threshold below which a cluster is considered controversial
        entropy_threshold: Entropy threshold above which a cluster is considered controversial
        max_discussion_rounds: Maximum number of discussion rounds for controversial clusters
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files
        verbose: Whether to print detailed logs
        consensus_model: Optional model specification for consensus checking and discussion.
            Can be a string (model name) or dict with 'provider' and 'model' keys.
            If not provided, picks from the user's available api_keys.
        base_urls: Custom base URLs for API endpoints. Can be:
                  - str: Single URL applied to all providers
                  - dict: Provider-specific URLs
        clusters_to_analyze: Optional list of cluster IDs to analyze. If provided,
            only the specified clusters will be processed. Cluster IDs must exist
            in the marker_genes dictionary. Non-existent cluster IDs will be
            ignored with a warning. If None (default), all clusters will be analyzed.
        force_rerun: If True, ignore cached results and force re-analysis of all
            specified clusters, including both the initial annotation phase and the
            discussion phase for controversial clusters. Useful when you want to
            re-analyze clusters with different context or for subtype identification.
            Default is False. Only effective when use_cache is True.
        request_timeout: Maximum read wait per HTTP request in seconds.
        max_runtime: Cooperative run budget in seconds, checked between requests.
        progress_callback: Optional callback receiving structured execution events.
        should_cancel: Optional callback checked before requests and stage transitions.
        prompt_template: Optional custom prompt template for the initial annotation
            phase. Supports the ``{species}``, ``{tissue}`` and ``{markers}``
            placeholders. If None (default), the built-in
            ``DEFAULT_PROMPT_TEMPLATE`` is used. Placed at the end of the
            signature to preserve compatibility with existing positional calls.

    Returns:
        dict[str, Any]: Dictionary containing consensus results and metadata

    """
    with annotation_execution(
        request_timeout=request_timeout,
        max_runtime=max_runtime,
        progress_callback=progress_callback,
        should_cancel=should_cancel,
    ) as execution:
        consensus_threshold = _normalize_probability(consensus_threshold, "consensus_threshold")
        entropy_threshold = _normalize_nonnegative_number(entropy_threshold, "entropy_threshold")
        max_discussion_rounds = _normalize_nonnegative_integer(
            max_discussion_rounds, "max_discussion_rounds"
        )
        use_cache = validate_bool(use_cache, "use_cache")
        force_rerun = validate_bool(force_rerun, "force_rerun")
        execution.use_cache = use_cache and not force_rerun
        verbose = validate_bool(verbose, "verbose")
        species = normalize_text(species, "species", required=True)
        tissue = normalize_text(tissue, "tissue")
        additional_context = normalize_text(additional_context, "additional_context")
        prompt_template = validate_prompt_template(prompt_template)

        metadata, marker_genes, resolved_models, api_keys, consensus_model_dict = (
            _prepare_interactive_annotation_context(
                marker_genes=marker_genes,
                species=species,
                tissue=tissue,
                models=models,
                api_keys=api_keys,
                consensus_model=consensus_model,
                clusters_to_analyze=clusters_to_analyze,
                consensus_threshold=consensus_threshold,
                entropy_threshold=entropy_threshold,
                max_discussion_rounds=max_discussion_rounds,
                verbose=verbose,
            )
        )

        model_results = _run_initial_annotations(
            marker_genes=marker_genes,
            species=species,
            models=resolved_models,
            api_keys=api_keys,
            tissue=tissue,
            additional_context=additional_context,
            prompt_template=prompt_template,
            use_cache=use_cache,
            force_rerun=force_rerun,
            cache_dir=cache_dir,
            base_urls=base_urls,
            verbose=verbose,
        )

        # Only models with usable initial evidence can participate in consensus.
        for key, labels in model_results.items():
            if not any(not is_unknown_annotation(label) for label in labels.values()):
                execution.unavailable[key] = "NoUsableAnnotations"
                execution.emit("model_failed", model=key, error="NoUsableAnnotations")
        model_results = {
            key: labels
            for key, labels in model_results.items()
            if any(not is_unknown_annotation(label) for label in labels.values())
        }
        active_models = [model for model in resolved_models if model.key in model_results]
        execution.consensus_candidates = [(model.provider, model.model) for model in active_models]
        metadata["model_failures"] = dict(execution.unavailable)
        metadata["successful_models"] = [model.key for model in active_models]
        if not model_results:
            write_log("No annotations were successful", level="error")
            error = "No annotations were successful"
            if "ProviderQuotaError" in execution.unavailable.values():
                error += ". Provider quota exhausted; check API balance or resource package."
            return _build_interactive_result(
                metadata=metadata,
                error=error,
            )

        if consensus_model_dict is None:
            consensus_model_dict = {
                "provider": active_models[0].provider,
                "model": active_models[0].model,
            }
        execution.phase = "consensus"
        execution.emit("consensus_started")
        # Check consensus
        consensus, consensus_proportion, entropy, controversial = check_consensus(
            model_results,
            consensus_threshold=consensus_threshold,
            entropy_threshold=entropy_threshold,
            api_keys=api_keys,
            consensus_model=consensus_model_dict,
            base_urls=base_urls,
        )

        if verbose:
            write_log(f"Found {len(controversial)} controversial clusters out of {len(consensus)}")

        resolved, discussion_logs, updated_cp, updated_h = (
            _resolve_controversial_clusters_if_needed(
                controversial=controversial,
                marker_genes=marker_genes,
                model_results=model_results,
                species=species,
                tissue=tissue,
                models=active_models,
                api_keys=api_keys,
                max_discussion_rounds=max_discussion_rounds,
                consensus_threshold=consensus_threshold,
                entropy_threshold=entropy_threshold,
                use_cache=use_cache,
                cache_dir=cache_dir,
                base_urls=base_urls,
                force_rerun=force_rerun,
                consensus_model=consensus_model_dict,
                verbose=verbose,
            )
        )
        _update_metrics_for_resolved_clusters(consensus_proportion, entropy, updated_cp, updated_h)
        if verbose and resolved:
            write_log(f"Successfully resolved {len(resolved)} controversial clusters")

        cleaned_annotations = _clean_annotations(_merge_consensus_and_resolved(consensus, resolved))

        metadata["model_failures"] = dict(execution.unavailable)
        execution.phase, execution.cluster, execution.round = "completed", None, None
        execution.emit("completed")
        return _build_interactive_result(
            metadata=metadata,
            consensus=cleaned_annotations,
            consensus_proportion=consensus_proportion,
            entropy=entropy,
            controversial_clusters=controversial,
            resolved=resolved,
            model_annotations=model_results,
            discussion_logs=discussion_logs,
        )


def _format_metadata_model_name(raw_model: Any) -> str:
    """Format one metadata model descriptor into a human-readable name."""
    if isinstance(raw_model, str):
        return raw_model
    if not isinstance(raw_model, dict):
        return str(raw_model)

    provider = str(raw_model.get("provider", "")).strip()
    model = str(raw_model.get("model", "")).strip()
    if provider and model:
        return f"{provider}:{model}"
    if model:
        return model
    if provider:
        return provider
    return str(raw_model)


def _format_metadata_model_names(raw_models: Any) -> list[str]:
    """Format model descriptors from metadata to human-readable names."""
    if raw_models is None:
        return []

    if isinstance(raw_models, (str, bytes, dict)):
        models_iterable = [raw_models]
    elif isinstance(raw_models, set):
        # Keep deterministic output for unordered containers.
        models_iterable = sorted(raw_models, key=lambda item: str(item))
    elif isinstance(raw_models, (list, tuple)):
        models_iterable = list(raw_models)
    else:
        try:
            models_iterable = list(raw_models)
        except TypeError:
            models_iterable = [raw_models]

    return [_format_metadata_model_name(model) for model in models_iterable]


def _display_metadata_value(value: Any) -> Any:
    """Normalize optional metadata values for report display."""
    if value is None:
        return "N/A"
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or "N/A"
    return value


_MISSING_REPORT_VALUE = object()


def _as_report_dict(value: Any) -> dict[Any, Any]:
    """Return mapping-shaped report data or an empty fallback."""
    return value if isinstance(value, dict) else {}


def _normalize_report_cluster_list(value: Any) -> list[str]:
    """Normalize report cluster collections with deterministic deduplication."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        if is_missing_value(item):
            continue
        cluster_id = str(item).strip()
        if cluster_id and cluster_id not in seen:
            seen.add(cluster_id)
            normalized.append(cluster_id)

    if isinstance(value, set):
        normalized.sort(key=cluster_sort_key)
    return normalized


def _get_report_cluster_value(
    values: dict[Any, Any],
    cluster_id: Any,
    default: Any = _MISSING_REPORT_VALUE,
) -> Any:
    """Read a cluster field using exact keys, then one unambiguous normalized key."""
    if cluster_id in values:
        return values[cluster_id]

    normalized_cluster_id = str(cluster_id).strip()
    matches = [
        value
        for raw_cluster_id, value in values.items()
        if str(raw_cluster_id).strip() == normalized_cluster_id
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        write_log(
            f"Ambiguous report values for normalized cluster ID '{normalized_cluster_id}'",
            level="warning",
        )
    return default


def _resolve_clusters_to_report(
    *,
    consensus: dict[str, Any],
    cluster_id: str | None,
) -> list[Any]:
    """Resolve report target clusters with deterministic ordering."""
    if cluster_id is not None:
        target_cluster = str(cluster_id).strip()
        if not target_cluster:
            return []
        for existing_cluster in consensus:
            if str(existing_cluster).strip() == target_cluster:
                return [existing_cluster]
        return []
    return sorted(consensus.keys(), key=lambda cid: cluster_sort_key(str(cid).strip()))


def _append_report_header(
    *,
    lines: list[str],
    separator: str,
    metadata: dict[str, Any],
) -> None:
    """Append report header section."""
    lines.append(separator)
    lines.append("MULTI-LLM CONSENSUS DISCUSSION REPORT")
    lines.append(separator)
    lines.append(f"Generated: {_display_metadata_value(metadata.get('timestamp'))}")
    lines.append(f"Species: {_display_metadata_value(metadata.get('species'))}")
    lines.append(f"Tissue: {_display_metadata_value(metadata.get('tissue'))}")
    model_names = _format_metadata_model_names(metadata.get("models", []))
    lines.append(f"Models: {', '.join(model_names) if model_names else 'N/A'}")
    lines.append(
        f"Consensus Threshold: {_display_metadata_value(metadata.get('consensus_threshold'))}"
    )
    lines.append(f"Entropy Threshold: {_display_metadata_value(metadata.get('entropy_threshold'))}")
    lines.append(
        f"Max Discussion Rounds: {_display_metadata_value(metadata.get('max_discussion_rounds'))}"
    )
    lines.append("")


def _append_initial_predictions_section(
    *,
    lines: list[str],
    model_annotations: dict[str, dict[str, Any]],
    cluster_id: Any,
) -> None:
    """Append initial prediction section for one cluster."""
    lines.append("")
    lines.append("-" * 40)
    lines.append("INITIAL PREDICTIONS")
    lines.append("-" * 40)

    for raw_model_name, predictions in model_annotations.items():
        model_name = str(raw_model_name)
        if not isinstance(predictions, dict):
            continue
        prediction = _get_report_cluster_value(predictions, cluster_id)
        if prediction is _MISSING_REPORT_VALUE:
            continue
        lines.append(f"\n[{model_name}]")
        lines.append(f"  {prediction}")


def _append_discussion_section(
    *,
    lines: list[str],
    discussion_logs: dict[str, list[dict[str, Any]]],
    cluster_id: Any,
) -> None:
    """Append discussion rounds section for one cluster."""
    rounds_raw = _get_report_cluster_value(discussion_logs, cluster_id, None)
    rounds: list[Any] = []
    if isinstance(rounds_raw, list):
        rounds = rounds_raw
    elif isinstance(rounds_raw, tuple):
        rounds = list(rounds_raw)

    if not rounds:
        lines.append("")
        lines.append("-" * 40)
        lines.append("NO DISCUSSION NEEDED")
        lines.append("-" * 40)
        lines.append("  Consensus reached with initial predictions.")
        return

    for round_idx, raw_round_responses in enumerate(rounds, start=1):
        round_responses = (
            raw_round_responses
            if isinstance(raw_round_responses, dict)
            else {"raw_response": raw_round_responses}
        )

        lines.append("")
        lines.append("-" * 40)
        lines.append(f"ROUND {round_idx} DISCUSSION")
        lines.append("-" * 40)

        for model_name, response in round_responses.items():
            lines.append(f"\n[{model_name}]")
            response_text = response if isinstance(response, str) else str(response)
            response_text = response_text.strip() or "Unknown"
            for line in response_text.split("\n"):
                lines.append(f"  {line}")


def _append_final_result_section(
    *,
    lines: list[str],
    cluster_id: Any,
    consensus: dict[str, Any],
    consensus_proportion: dict[str, Any],
    entropy: dict[str, Any],
    controversial_clusters: list[str],
) -> None:
    """Append final result section for one cluster."""
    normalized_cluster_id = str(cluster_id).strip()
    controversial_cluster_set = {str(cid).strip() for cid in controversial_clusters}

    lines.append("")
    lines.append("-" * 40)
    lines.append("FINAL RESULT")
    lines.append("-" * 40)
    lines.append(f"  Final Annotation: {_get_report_cluster_value(consensus, cluster_id, 'N/A')}")
    lines.append(
        "  Consensus Proportion: "
        f"{_get_report_cluster_value(consensus_proportion, cluster_id, 'N/A')}"
    )
    lines.append(f"  Entropy: {_get_report_cluster_value(entropy, cluster_id, 'N/A')}")
    lines.append(
        f"  Was Controversial: {'Yes' if normalized_cluster_id in controversial_cluster_set else 'No'}"
    )
    lines.append("")


def format_discussion_report(
    results: dict[str, Any],
    cluster_id: str | None = None,
    output_file: str | None = None,
) -> str:
    """Format discussion results into a clean, readable report.

    This function generates a structured report showing:
    1. Initial predictions from each model
    2. Full discussion for each round
    3. Final consensus result

    Args:
        results: The results dictionary from interactive_consensus_annotation
        cluster_id: Optional cluster ID to filter. If None, reports all clusters.
        output_file: Optional file path to save the report. If None, only returns string.

    Returns:
        str: Formatted discussion report

    Example:
        >>> results = interactive_consensus_annotation(...)
        >>> report = format_discussion_report(results, cluster_id="0")
        >>> print(report)

        # Or save to file:
        >>> format_discussion_report(results, output_file="discussion_report.txt")
    """
    if not isinstance(results, dict):
        raise ValueError(f"results must be a dict, got {type(results).__name__}")

    lines: list[str] = []
    sep = "=" * 80

    model_annotations = _as_report_dict(results.get("model_annotations", {}))
    discussion_logs = _as_report_dict(results.get("discussion_logs", {}))
    consensus = _as_report_dict(results.get("consensus", {}))
    consensus_proportion = _as_report_dict(results.get("consensus_proportion", {}))
    entropy = _as_report_dict(results.get("entropy", {}))
    controversial_clusters = _normalize_report_cluster_list(
        results.get("controversial_clusters", [])
    )
    metadata = _as_report_dict(results.get("metadata", {}))

    _append_report_header(lines=lines, separator=sep, metadata=metadata)

    clusters_to_report = _resolve_clusters_to_report(consensus=consensus, cluster_id=cluster_id)

    for cid in clusters_to_report:
        lines.append(sep)
        lines.append(f"CLUSTER {cid}")
        lines.append(sep)
        _append_initial_predictions_section(
            lines=lines,
            model_annotations=model_annotations,
            cluster_id=cid,
        )
        _append_discussion_section(
            lines=lines,
            discussion_logs=discussion_logs,
            cluster_id=cid,
        )
        _append_final_result_section(
            lines=lines,
            cluster_id=cid,
            consensus=consensus,
            consensus_proportion=consensus_proportion,
            entropy=entropy,
            controversial_clusters=controversial_clusters,
        )

    lines.append(sep)
    lines.append("END OF REPORT")
    lines.append(sep)

    report = "\n".join(lines)

    # Save to file if requested
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        write_log(f"Discussion report saved to: {output_file}")

    return report
