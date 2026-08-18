"""Validation for annotation API requests."""

import math
from dataclasses import dataclass
from typing import Any


_FALLBACK_PROVIDERS = frozenset(
    {
        "anthropic",
        "deepseek",
        "gemini",
        "grok",
        "kimi",
        "minimax",
        "openai",
        "openrouter",
        "qwen",
        "stepfun",
        "zhipu",
    }
)

try:
    from mllmcelltype.config import get_supported_providers
except ImportError:
    SUPPORTED_PROVIDERS = _FALLBACK_PROVIDERS
else:
    SUPPORTED_PROVIDERS = frozenset(get_supported_providers())

MAX_MODELS_PER_TASK = 30
MAX_API_KEY_LENGTH = 8192


class AnnotationValidationError(ValueError):
    """Raised when an annotation request violates the public API contract."""


@dataclass(frozen=True)
class AnnotationRequest:
    """Validated annotation parameters."""

    task_id: str
    species: str
    tissue: str
    models: list[str]
    api_keys: dict[str, str]
    consensus_threshold: float
    entropy_threshold: float
    max_rounds: int
    consensus_model: str | None


def _bounded_text(value: Any, field: str, *, required: bool, max_length: int) -> str:
    if not isinstance(value, str):
        raise AnnotationValidationError(f"{field} must be a string")
    cleaned = value.strip()
    if required and not cleaned:
        raise AnnotationValidationError(f"{field} is required")
    if len(cleaned) > max_length:
        raise AnnotationValidationError(
            f"{field} must be at most {max_length} characters"
        )
    return cleaned


def _bounded_float(
    value: Any,
    field: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        raise AnnotationValidationError(f"{field} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError(f"{field} must be a number") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise AnnotationValidationError(
            f"{field} must be between {minimum} and {maximum}"
        )
    return parsed


def _normalize_model(value: Any, field: str) -> tuple[str, str]:
    model = _bounded_text(value, field, required=True, max_length=300)
    if ":" not in model:
        raise AnnotationValidationError(
            "Each model must use the 'provider:model' format"
        )
    raw_provider, raw_model_name = model.split(":", 1)
    provider = raw_provider.strip().lower()
    model_name = raw_model_name.strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise AnnotationValidationError(f"Unsupported model provider: {provider}")
    if not model_name:
        raise AnnotationValidationError("Model name must not be empty")
    return f"{provider}:{model_name}", provider


def parse_annotation_request(data: Any) -> AnnotationRequest:
    """Validate and normalize the JSON body for ``POST /api/annotate``."""
    if not isinstance(data, dict):
        raise AnnotationValidationError("Request body must be a JSON object")

    task_id = _bounded_text(
        data.get("task_id"), "task_id", required=True, max_length=64
    )
    species = _bounded_text(
        data.get("species", "human"), "species", required=True, max_length=100
    )
    tissue = _bounded_text(
        data.get("tissue", ""), "tissue", required=False, max_length=200
    )

    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise AnnotationValidationError("At least one model must be selected")
    if len(raw_models) > MAX_MODELS_PER_TASK:
        raise AnnotationValidationError(
            f"No more than {MAX_MODELS_PER_TASK} models may be selected"
        )

    models: list[str] = []
    providers: set[str] = set()
    for raw_model in raw_models:
        model, provider = _normalize_model(raw_model, "Each model")
        if model in models:
            raise AnnotationValidationError(f"Duplicate model selection: {model}")
        models.append(model)
        providers.add(provider)

    raw_api_keys = data.get("api_keys")
    if not isinstance(raw_api_keys, dict):
        raise AnnotationValidationError("api_keys must be an object")

    api_keys: dict[str, str] = {}
    for provider in providers:
        api_key = _bounded_text(
            raw_api_keys.get(provider),
            f"API key for {provider}",
            required=True,
            max_length=MAX_API_KEY_LENGTH,
        )
        api_keys[provider] = api_key

    consensus_threshold = _bounded_float(
        data.get("consensusThreshold", 0.7),
        "consensusThreshold",
        minimum=0.1,
        maximum=1.0,
    )
    entropy_threshold = _bounded_float(
        data.get("entropyThreshold", 1.0),
        "entropyThreshold",
        minimum=0.1,
        maximum=2.0,
    )

    raw_max_rounds = data.get("maxDiscussionRounds", 3)
    if isinstance(raw_max_rounds, bool):
        raise AnnotationValidationError("maxDiscussionRounds must be an integer")
    try:
        numeric_max_rounds = float(raw_max_rounds)
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError(
            "maxDiscussionRounds must be an integer"
        ) from exc
    if not math.isfinite(numeric_max_rounds) or not numeric_max_rounds.is_integer():
        raise AnnotationValidationError("maxDiscussionRounds must be an integer")
    max_rounds = int(numeric_max_rounds)
    if not 1 <= max_rounds <= 5:
        raise AnnotationValidationError("maxDiscussionRounds must be between 1 and 5")

    raw_consensus_model = data.get("consensusModel")
    consensus_model = None
    if raw_consensus_model not in (None, ""):
        consensus_model, _ = _normalize_model(raw_consensus_model, "consensusModel")
        if consensus_model not in models:
            raise AnnotationValidationError(
                "consensusModel must be one of the selected models"
            )

    return AnnotationRequest(
        task_id=task_id,
        species=species,
        tissue=tissue,
        models=models,
        api_keys=api_keys,
        consensus_threshold=consensus_threshold,
        entropy_threshold=entropy_threshold,
        max_rounds=max_rounds,
        consensus_model=consensus_model,
    )
