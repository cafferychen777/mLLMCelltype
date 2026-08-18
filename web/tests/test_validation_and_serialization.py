"""Tests for public request validation and strict JSON conversion."""

import math

import numpy as np
import pytest

from utils.api_validator import _sdk_error
from utils.serialization import to_json_compatible
from utils.task_validation import AnnotationValidationError, parse_annotation_request


def valid_request() -> dict:
    return {
        "task_id": "task-1",
        "species": "human",
        "tissue": "blood",
        "models": ["openai:gpt-4.1", "anthropic:claude-sonnet-4-5"],
        "api_keys": {"openai": "key-a", "anthropic": "key-b"},
        "consensusThreshold": 0.7,
        "entropyThreshold": 1.0,
        "maxDiscussionRounds": 3,
        "consensusModel": "openai:gpt-4.1",
    }


def test_valid_annotation_request_is_normalized() -> None:
    request = parse_annotation_request(valid_request())

    assert request.models == [
        "openai:gpt-4.1",
        "anthropic:claude-sonnet-4-5",
    ]
    assert request.max_rounds == 3
    assert request.consensus_model == "openai:gpt-4.1"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("consensusThreshold", "nan", "between"),
        ("entropyThreshold", math.inf, "between"),
        ("maxDiscussionRounds", 2.5, "integer"),
        ("maxDiscussionRounds", 6, "between"),
    ],
)
def test_invalid_numeric_parameters_are_rejected(
    field: str, value: object, message: str
) -> None:
    payload = valid_request()
    payload[field] = value

    with pytest.raises(AnnotationValidationError, match=message):
        parse_annotation_request(payload)


def test_missing_provider_key_is_rejected() -> None:
    payload = valid_request()
    del payload["api_keys"]["anthropic"]

    with pytest.raises(AnnotationValidationError, match="API key for anthropic"):
        parse_annotation_request(payload)


def test_strict_json_conversion_handles_scientific_values() -> None:
    converted = to_json_compatible(
        {
            "array": np.array([1.0, np.nan]),
            "scalar": np.float64(2.5),
            "infinite": math.inf,
            "tuple": ("a", 1),
        }
    )

    assert converted == {
        "array": [1.0, None],
        "scalar": 2.5,
        "infinite": None,
        "tuple": ["a", 1],
    }


def test_provider_errors_are_bounded_and_redact_credentials() -> None:
    api_key = "secret-provider-key"
    error = RuntimeError(f"Request failed for {api_key}: " + "x" * 1000)

    message = _sdk_error(error, api_key)

    assert api_key not in message
    assert "[redacted]" in message
    assert len(message) <= 511
