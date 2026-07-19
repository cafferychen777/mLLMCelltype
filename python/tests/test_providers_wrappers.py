"""Tests for provider wrapper functions (request construction and routing)."""

from __future__ import annotations

import builtins
import json
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mllmcelltype.providers.anthropic import (
    _parse_anthropic_response,
    extract_anthropic_usage,
    process_anthropic,
)
from mllmcelltype.providers.common import (
    NonRetryableProviderError,
    RetryableProviderError,
    parse_chat_completions_response,
)
from mllmcelltype.providers.deepseek import process_deepseek
from mllmcelltype.providers.gemini import extract_gemini_usage, process_gemini
from mllmcelltype.providers.grok import process_grok
from mllmcelltype.providers.minimax import (
    _parse_minimax_response,
    process_minimax,
)
from mllmcelltype.providers.openai import process_openai
from mllmcelltype.providers.openrouter import process_openrouter
from mllmcelltype.providers.stepfun import process_stepfun
from mllmcelltype.providers.zhipu import process_zhipu


class _FakeGoogleAPIError(Exception):
    """Minimal Google SDK API error carrying an HTTP-like status code."""

    def __init__(self, code: int, message: str):
        self.code = code
        super().__init__(message)


def _build_fake_google_modules(
    client: object,
    config_factory: MagicMock | None = None,
    http_options_factory: MagicMock | None = None,
) -> dict[str, ModuleType]:
    """Build fake google-genai modules for deterministic Gemini tests."""
    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    errors_module = ModuleType("google.genai.errors")
    types_module = ModuleType("google.genai.types")

    if config_factory is None:
        config_factory = MagicMock(return_value={"temperature": 0.7, "max_output_tokens": 4096})
    if http_options_factory is None:
        http_options_factory = MagicMock(side_effect=lambda **kwargs: kwargs)

    genai_module.Client = MagicMock(return_value=client)  # type: ignore[attr-defined]
    errors_module.APIError = _FakeGoogleAPIError  # type: ignore[attr-defined]
    types_module.GenerateContentConfig = config_factory  # type: ignore[attr-defined]
    types_module.HttpOptions = http_options_factory  # type: ignore[attr-defined]
    genai_module.errors = errors_module  # type: ignore[attr-defined]
    genai_module.types = types_module  # type: ignore[attr-defined]
    google_module.genai = genai_module  # type: ignore[attr-defined]

    return {
        "google": google_module,
        "google.genai": genai_module,
        "google.genai.errors": errors_module,
        "google.genai.types": types_module,
    }


@patch("mllmcelltype.providers.deepseek.call_openai_compatible_api")
@patch("mllmcelltype.providers.deepseek.resolve_endpoint_url")
def test_process_deepseek_request_shape(mock_resolve_endpoint, mock_call_api):
    """Test DeepSeek wrapper sends expected body and retry parameters."""
    mock_resolve_endpoint.return_value = "https://deepseek.example/v1"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_deepseek("genes", "deepseek-v4-flash", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["provider_name"] == "DeepSeek"
    assert kwargs["url"] == "https://deepseek.example/v1"
    assert kwargs["request_json"] is True
    assert kwargs["max_retries"] == 5
    assert kwargs["retry_delay"] == 3
    assert kwargs["timeout"] == 90
    assert kwargs["body"]["temperature"] == 0.7
    assert kwargs["body"]["max_tokens"] == 4096


@patch("mllmcelltype.providers.grok.call_openai_compatible_api")
@patch("mllmcelltype.providers.grok.resolve_endpoint_url")
def test_process_grok_request_shape(mock_resolve_endpoint, mock_call_api):
    """Test Grok wrapper builds OpenAI-compatible body."""
    mock_resolve_endpoint.return_value = "https://grok.example/v1"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_grok("genes", "grok-4.3", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["provider_name"] == "Grok"
    assert kwargs["body"]["model"] == "grok-4.3"
    assert kwargs["body"]["messages"][0]["content"] == "genes"


@patch("mllmcelltype.providers.openai.call_openai_compatible_api")
@patch("mllmcelltype.providers.openai.resolve_endpoint_url")
def test_process_openai_request_shape(mock_resolve_endpoint, mock_call_api):
    """Test OpenAI wrapper uses normalized shared chat payload format."""
    mock_resolve_endpoint.return_value = "https://openai.example/v1"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_openai("genes", "gpt-5.5", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["provider_name"] == "OpenAI"
    assert kwargs["body"]["model"] == "gpt-5.5"
    assert kwargs["body"]["messages"] == [{"role": "user", "content": "genes"}]


@patch("mllmcelltype.providers.openrouter.write_log")
@patch("mllmcelltype.providers.openrouter.call_openai_compatible_api")
@patch("mllmcelltype.providers.openrouter.resolve_endpoint_url")
def test_process_openrouter_warns_for_non_provider_model(
    mock_resolve_endpoint,
    mock_call_api,
    mock_write_log,
):
    """Test OpenRouter warns when model lacks provider/model format."""
    mock_resolve_endpoint.return_value = "https://openrouter.example/v1"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_openrouter("genes", "gpt-5.5", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["extra_headers"]["X-Title"] == "mLLMCelltype"
    assert any(
        "may not be in the correct format" in call.args[0] and call.kwargs.get("level") == "warning"
        for call in mock_write_log.call_args_list
    )


@patch("mllmcelltype.providers.stepfun.call_openai_compatible_api")
@patch("mllmcelltype.providers.stepfun.resolve_endpoint_url")
def test_process_stepfun_request_shape(mock_resolve_endpoint, mock_call_api):
    """Test StepFun wrapper includes expected generation controls."""
    mock_resolve_endpoint.return_value = "https://stepfun.example/v1"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_stepfun("genes", "step-3.5-flash", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["provider_name"] == "StepFun"
    assert kwargs["body"]["temperature"] == 0.7
    assert kwargs["body"]["max_tokens"] == 4096


@patch("mllmcelltype.providers.zhipu.call_openai_compatible_api")
@patch("mllmcelltype.providers.zhipu.resolve_endpoint_url")
def test_process_zhipu_request_shape(mock_resolve_endpoint, mock_call_api):
    """Test Zhipu wrapper includes expected generation controls."""
    mock_resolve_endpoint.return_value = "https://zhipu.example/v1"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_zhipu("genes", "glm-5.1", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["provider_name"] == "Zhipu AI"
    assert kwargs["body"]["temperature"] == 0.7
    assert kwargs["body"]["max_tokens"] == 4096


@patch("mllmcelltype.providers.anthropic.call_http_api_with_retry")
@patch("mllmcelltype.providers.anthropic.resolve_endpoint_url")
def test_process_anthropic_alias_resolution_and_headers(
    mock_resolve_endpoint,
    mock_call_http,
):
    """Test Anthropic wrapper resolves aliases and sends expected headers."""
    mock_resolve_endpoint.return_value = "https://anthropic.example/v1/messages"
    mock_call_http.return_value = ["Cluster 1: T cells"]

    result = process_anthropic("genes", "claude-opus-latest", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_http.call_args.kwargs
    assert kwargs["body"]["model"] == "claude-opus-4-7"
    assert kwargs["headers"]["x-api-key"] == "test-key"
    assert kwargs["headers"]["anthropic-version"] == "2023-06-01"


def test_parse_anthropic_response_rejects_non_string():
    """Test Anthropic parser rejects non-string text payload."""
    payload = {"content": [{"text": 123}]}
    with pytest.raises(NonRetryableProviderError, match="non-string"):
        _parse_anthropic_response(payload)


def test_parse_anthropic_response_concatenates_text_blocks_and_skips_thinking():
    """Test the parser keeps every text block and ignores a leading thinking block."""
    payload = {
        "content": [
            {"type": "thinking", "thinking": "reasoning..."},
            {"type": "text", "text": "Cluster 1: T cells"},
            {"type": "text", "text": "Cluster 2: B cells"},
        ]
    }
    assert _parse_anthropic_response(payload) == [
        "Cluster 1: T cells",
        "Cluster 2: B cells",
    ]


def test_parse_anthropic_response_rejects_thinking_only_response():
    """Test a response carrying only non-text blocks is rejected."""
    payload = {"content": [{"type": "thinking", "thinking": "reasoning..."}]}
    with pytest.raises(ValueError, match="no text block found"):
        _parse_anthropic_response(payload)


def test_parse_chat_completions_warns_on_length_truncation():
    """Test a length-truncated OpenAI-compatible response logs a warning."""
    content = {"choices": [{"message": {"content": "T cells"}, "finish_reason": "length"}]}
    with patch("mllmcelltype.providers.common.write_log") as mock_log:
        assert parse_chat_completions_response(content, "OpenAI") == ["T cells"]
    assert any("truncated" in str(call.args[0]) for call in mock_log.call_args_list)


def test_parse_chat_completions_no_warning_on_normal_stop():
    """Test a normally-finished response does not emit a truncation warning."""
    content = {"choices": [{"message": {"content": "T cells"}, "finish_reason": "stop"}]}
    with patch("mllmcelltype.providers.common.write_log") as mock_log:
        parse_chat_completions_response(content, "OpenAI")
    assert not any("truncated" in str(call.args[0]) for call in mock_log.call_args_list)


def test_parse_anthropic_response_warns_on_max_tokens_truncation():
    """Test a max_tokens-truncated Anthropic response logs a warning."""
    content = {
        "content": [{"type": "text", "text": "T cells"}],
        "stop_reason": "max_tokens",
    }
    with patch("mllmcelltype.providers.anthropic.write_log") as mock_log:
        assert _parse_anthropic_response(content) == ["T cells"]
    assert any("truncated" in str(call.args[0]) for call in mock_log.call_args_list)


def test_parse_anthropic_response_no_warning_on_normal_stop():
    """Test a normally-finished Anthropic response emits no truncation warning."""
    content = {
        "content": [{"type": "text", "text": "T cells"}],
        "stop_reason": "end_turn",
    }
    with patch("mllmcelltype.providers.anthropic.write_log") as mock_log:
        _parse_anthropic_response(content)
    assert not any("truncated" in str(call.args[0]) for call in mock_log.call_args_list)


def test_extract_anthropic_usage_normalizes_token_fields():
    """Test Anthropic usage fields map to the shared provider schema."""
    usage = extract_anthropic_usage({"usage": {"input_tokens": 12, "output_tokens": 5}})

    assert usage == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}


def test_extract_anthropic_usage_partial_counts_do_not_invent_total():
    """Test incomplete Anthropic usage preserves missing values without guessing."""
    usage = extract_anthropic_usage({"usage": {"input_tokens": 12}})

    assert usage == {"prompt_tokens": 12, "completion_tokens": None, "total_tokens": None}


def test_extract_anthropic_usage_rejects_invalid_counts():
    """Test Anthropic usage follows the shared strict count contract."""
    usage = extract_anthropic_usage({"usage": {"input_tokens": -1, "output_tokens": True}})

    assert usage == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


@patch("mllmcelltype.providers.minimax.call_openai_compatible_api")
@patch("mllmcelltype.providers.minimax.get_working_minimax_endpoint")
def test_process_minimax_uses_smart_endpoint_by_default(
    mock_get_working_endpoint,
    mock_call_api,
):
    """Test MiniMax wrapper uses smart endpoint selection when base_url is absent."""
    mock_get_working_endpoint.return_value = "https://api.minimax.io/v1/chat/completions"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_minimax("genes", "MiniMax-M2.7", "test-key")

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["url"] == "https://api.minimax.io/v1/chat/completions"
    assert kwargs["non_retry_exceptions"] == (NonRetryableProviderError,)
    assert kwargs["body"]["messages"] == [{"role": "user", "name": "user", "content": "genes"}]


@patch("mllmcelltype.providers.minimax.call_openai_compatible_api")
@patch("mllmcelltype.providers.minimax.resolve_endpoint_url")
def test_process_minimax_uses_custom_base_url(
    mock_resolve_endpoint,
    mock_call_api,
):
    """Test MiniMax wrapper uses explicit base_url when provided."""
    mock_resolve_endpoint.return_value = "https://custom.example/v1/chat/completions"
    mock_call_api.return_value = ["Cluster 1: T cells"]

    result = process_minimax(
        "genes",
        "MiniMax-M2.7",
        "test-key",
        base_url="https://custom.example/v1/chat/completions",
    )

    assert result == ["Cluster 1: T cells"]
    kwargs = mock_call_api.call_args.kwargs
    assert kwargs["url"] == "https://custom.example/v1/chat/completions"


def test_parse_minimax_response_strips_think_block():
    """Test MiniMax parser removes reasoning block and trailing commas."""
    payload = {
        "choices": [{"message": {"content": "<think>reasoning</think>\nCluster 1: T cells,\n"}}]
    }
    assert _parse_minimax_response(payload) == ["Cluster 1: T cells"]


def test_parse_minimax_response_rejects_non_string_content():
    """Test MiniMax parser rejects non-string message content."""
    payload = {
        "choices": [{"message": {"content": 123}}],
    }
    with pytest.raises(NonRetryableProviderError, match="non-string"):
        _parse_minimax_response(payload)


def test_parse_minimax_response_invalid_shape_raises_value_error():
    """Test MiniMax parser raises clear error for invalid payload shape."""
    with pytest.raises(ValueError, match="Unexpected response format from MiniMax"):
        _parse_minimax_response({"invalid": "payload"})


def test_parse_minimax_response_permanent_business_error_is_non_retryable():
    """Test a fatal MiniMax business error (auth) fails fast without retry."""
    payload = {
        "base_resp": {"status_code": 1004, "status_msg": "not authorized"},
        "choices": [{"message": {"content": "ignored"}}],
    }
    with pytest.raises(NonRetryableProviderError, match="MiniMax API error"):
        _parse_minimax_response(payload)


def test_parse_minimax_response_transient_business_error_is_retryable():
    """Test a transient MiniMax business error (rate limit) is retryable.

    MiniMax reports rate limits/timeouts via HTTP 200 + base_resp.status_code;
    these must be retried instead of dropping the model from the panel.
    """
    payload = {
        "base_resp": {"status_code": 1002, "status_msg": "rate limit"},
        "choices": [{"message": {"content": "ignored"}}],
    }
    with pytest.raises(RetryableProviderError, match="MiniMax API error"):
        _parse_minimax_response(payload)


@patch("mllmcelltype.providers.common.time.sleep")
@patch("mllmcelltype.providers.minimax.requests.post")
@patch("mllmcelltype.providers.minimax.get_working_minimax_endpoint")
def test_process_minimax_retries_transient_business_error(
    mock_endpoint, mock_post, mock_sleep
):
    """Test a transient base_resp error is retried through the shared backoff."""
    mock_endpoint.return_value = "https://api.minimax.io/v1/chat/completions"
    transient = MagicMock(status_code=200)
    transient.json.return_value = {
        "base_resp": {"status_code": 1002, "status_msg": "rate limit"}
    }
    success = MagicMock(status_code=200)
    success.json.return_value = {
        "base_resp": {"status_code": 0},
        "choices": [{"message": {"content": "Cluster 1: T cells"}}],
    }
    mock_post.side_effect = [transient, success]

    result = process_minimax("genes", "MiniMax-M2.7", "test-key")

    assert result == ["Cluster 1: T cells"]
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once_with(2)


def test_retryable_provider_error_is_recoverable_by_consensus():
    """Test RetryableProviderError is caught by the consensus recovery net.

    A transient error that outlives all retries must drop only its own model,
    not abort the whole consensus run, so it has to be a member of
    RECOVERABLE_LLM_EXCEPTIONS.
    """
    from mllmcelltype.consensus import RECOVERABLE_LLM_EXCEPTIONS

    assert isinstance(RetryableProviderError("boom"), RECOVERABLE_LLM_EXCEPTIONS)


@patch("mllmcelltype.providers.common.time.sleep")
@patch("mllmcelltype.providers.minimax.requests.post")
@patch("mllmcelltype.providers.minimax.get_working_minimax_endpoint")
def test_process_minimax_persistent_transient_raises_recoverable(
    mock_endpoint, mock_post, mock_sleep
):
    """Test a transient error on every attempt exhausts retries and raises a
    recoverable RetryableProviderError instead of a bare Exception."""
    from mllmcelltype.consensus import RECOVERABLE_LLM_EXCEPTIONS

    mock_endpoint.return_value = "https://api.minimax.io/v1/chat/completions"
    transient = MagicMock(status_code=200)
    transient.json.return_value = {
        "base_resp": {"status_code": 1002, "status_msg": "rate limit"}
    }
    mock_post.return_value = transient

    with pytest.raises(RetryableProviderError) as excinfo:
        process_minimax("genes", "MiniMax-M2.7", "test-key")

    assert mock_post.call_count == 3
    assert isinstance(excinfo.value, RECOVERABLE_LLM_EXCEPTIONS)


def test_parse_minimax_response_string_status_code_is_coerced():
    """Test a transient status code delivered as a string is coerced and retried."""
    payload = {
        "base_resp": {"status_code": "1002", "status_msg": "rate limit"},
        "choices": [{"message": {"content": "ignored"}}],
    }
    with pytest.raises(RetryableProviderError, match="MiniMax API error"):
        _parse_minimax_response(payload)


@patch("mllmcelltype.providers.gemini.time.sleep")
def test_process_gemini_retries_then_succeeds(mock_sleep):
    """Test Gemini retries transient errors and succeeds on a later attempt."""
    fake_response = SimpleNamespace(text="Cluster 1: T cells,\nCluster 2: B cells,")
    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=MagicMock()))
    fake_client.models.generate_content.side_effect = [
        _FakeGoogleAPIError(429, "transient"),
        fake_response,
    ]

    fake_modules = _build_fake_google_modules(client=fake_client)

    with patch.dict("sys.modules", fake_modules, clear=False):
        result = process_gemini("genes", "gemini-3.1-pro-preview", "test-key")

    assert result == ["Cluster 1: T cells", "Cluster 2: B cells"]
    assert fake_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once_with(2)


def test_process_gemini_non_string_text_is_rejected():
    """Test Gemini rejects non-string response text content."""
    fake_response = SimpleNamespace(text=123)
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=MagicMock(return_value=fake_response))
    )

    fake_modules = _build_fake_google_modules(client=fake_client)

    with (
        patch.dict("sys.modules", fake_modules, clear=False),
        pytest.raises(NonRetryableProviderError, match="non-string"),
    ):
        process_gemini("genes", "gemini-3.1-pro-preview", "test-key")


def test_process_gemini_trims_api_key_for_client():
    """Test Gemini passes normalized API key to the SDK client."""
    fake_response = SimpleNamespace(text="Cluster 1: T cells")
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=MagicMock(return_value=fake_response))
    )

    fake_modules = _build_fake_google_modules(client=fake_client)

    with patch.dict("sys.modules", fake_modules, clear=False):
        result = process_gemini("genes", "gemini-3.1-pro-preview", "  test-key  ")

    assert result == ["Cluster 1: T cells"]
    client_kwargs = fake_modules["google.genai"].Client.call_args.kwargs
    assert client_kwargs["api_key"] == "test-key"
    assert client_kwargs["http_options"] == {"timeout": 30_000}


def test_process_gemini_cleans_blank_lines_whitespace_and_commas():
    """Test Gemini response cleanup keeps only cleaned annotation lines."""
    fake_response = SimpleNamespace(text="  Cluster 1: T cells,\n\n Cluster 2: B cells,  \n   ")
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=MagicMock(return_value=fake_response))
    )

    fake_modules = _build_fake_google_modules(client=fake_client)

    with patch.dict("sys.modules", fake_modules, clear=False):
        result = process_gemini("genes", "gemini-3.1-pro-preview", "test-key")

    assert result == ["Cluster 1: T cells", "Cluster 2: B cells"]


@patch("mllmcelltype.providers.gemini.time.sleep")
def test_process_gemini_missing_text_fails_fast_without_retry(mock_sleep):
    """Test missing response text raises ValueError immediately (no useless retries)."""
    fake_response = SimpleNamespace(text=None)
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=MagicMock(return_value=fake_response))
    )

    fake_modules = _build_fake_google_modules(client=fake_client)

    with (
        patch.dict("sys.modules", fake_modules, clear=False),
        pytest.raises(ValueError, match="missing text content"),
    ):
        process_gemini("genes", "gemini-3.1-pro-preview", "test-key")

    assert fake_client.models.generate_content.call_count == 1
    mock_sleep.assert_not_called()


def test_process_gemini_import_error_message_is_clear():
    """Test missing google-genai dependency raises actionable guidance."""
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google" or name.startswith("google.genai"):
            raise ImportError("no module named google")
        return original_import(name, globals, locals, fromlist, level)

    with (
        patch("builtins.__import__", side_effect=_fake_import),
        pytest.raises(ImportError, match="google-genai"),
    ):
        process_gemini("genes", "gemini-3.1-pro-preview", "test-key")


@patch("mllmcelltype.providers.gemini.write_log")
def test_process_gemini_honors_custom_base_url(mock_write_log):
    """Test Gemini routes custom endpoints through SDK HTTP options."""
    fake_response = SimpleNamespace(text="Cluster 1: T cells")
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=MagicMock(return_value=fake_response))
    )
    http_options_factory = MagicMock(return_value="configured-http-options")
    fake_modules = _build_fake_google_modules(
        client=fake_client,
        http_options_factory=http_options_factory,
    )

    with patch.dict("sys.modules", fake_modules, clear=False):
        result = process_gemini(
            "genes",
            "gemini-3.1-pro-preview",
            "test-key",
            base_url="  https://proxy.example/v1/  ",
        )

    assert result == ["Cluster 1: T cells"]
    http_options_factory.assert_called_once_with(
        timeout=30_000,
        base_url="https://proxy.example/v1",
    )
    fake_modules["google.genai"].Client.assert_called_once_with(
        api_key="test-key",
        http_options="configured-http-options",
    )
    assert any(
        "Using custom base URL: https://proxy.example/v1" in call.args[0]
        for call in mock_write_log.call_args_list
    )


@patch("mllmcelltype.providers.gemini.write_log")
def test_process_gemini_whitespace_base_url_uses_default_route(mock_write_log):
    """Test blank base_url values do not create a custom SDK route."""
    fake_response = SimpleNamespace(text="Cluster 1: T cells")
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=MagicMock(return_value=fake_response))
    )
    http_options_factory = MagicMock(return_value="default-http-options")
    fake_modules = _build_fake_google_modules(
        client=fake_client,
        http_options_factory=http_options_factory,
    )

    with patch.dict("sys.modules", fake_modules, clear=False):
        result = process_gemini(
            "genes",
            "gemini-3.1-pro-preview",
            "test-key",
            base_url="   ",
        )

    assert result == ["Cluster 1: T cells"]
    http_options_factory.assert_called_once_with(timeout=30_000)
    assert not any(
        "Using custom base URL" in call.args[0] for call in mock_write_log.call_args_list
    )


def test_process_gemini_rejects_invalid_custom_base_url_before_client_creation():
    """Test invalid Gemini routing fails before initializing the SDK client."""
    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=MagicMock()))
    fake_modules = _build_fake_google_modules(client=fake_client)

    with (
        patch.dict("sys.modules", fake_modules, clear=False),
        pytest.raises(ValueError, match="Invalid base URL"),
    ):
        process_gemini(
            "genes",
            "gemini-3.1-pro-preview",
            "test-key",
            base_url="not-a-url",
        )

    fake_modules["google.genai"].Client.assert_not_called()


def _ok_response(payload: dict) -> MagicMock:
    """Build a 200 OK mock requests.Response returning ``payload`` from .json()."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    return response


@patch("mllmcelltype.providers.openai.requests.post")
@patch("mllmcelltype.providers.openai.resolve_endpoint_url")
def test_process_openai_surfaces_usage_through_sink(mock_resolve_endpoint, mock_post):
    """Test an OpenAI-compatible provider fills the caller's usage sink end-to-end."""
    mock_resolve_endpoint.return_value = "https://openai.example/v1"
    mock_post.return_value = _ok_response(
        {
            "choices": [{"message": {"content": "Cluster 1: T cells"}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36},
        }
    )
    sink: dict = {}

    result = process_openai("genes", "gpt-5.5", "test-key", usage_sink=sink)

    assert result == ["Cluster 1: T cells"]
    assert sink == {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36}


@patch("mllmcelltype.providers.anthropic.requests.post")
@patch("mllmcelltype.providers.anthropic.resolve_endpoint_url")
def test_process_anthropic_surfaces_usage_through_sink(mock_resolve_endpoint, mock_post):
    """Test Anthropic follows the same optional usage contract as other providers."""
    mock_resolve_endpoint.return_value = "https://anthropic.example/v1/messages"
    mock_post.return_value = _ok_response(
        {
            "content": [{"text": "Cluster 1: T cells"}],
            "usage": {"input_tokens": 20, "output_tokens": 4},
        }
    )
    sink: dict = {}

    result = process_anthropic(
        "genes",
        "claude-opus-4-7",
        "test-key",
        usage_sink=sink,
    )

    assert result == ["Cluster 1: T cells"]
    assert sink == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}


@patch("mllmcelltype.providers.openai.requests.post")
@patch("mllmcelltype.providers.openai.resolve_endpoint_url")
def test_process_openai_default_path_unchanged(mock_resolve_endpoint, mock_post):
    """Test default invocation (no sink) returns exactly what it does today."""
    mock_resolve_endpoint.return_value = "https://openai.example/v1"
    mock_post.return_value = _ok_response(
        {
            "choices": [{"message": {"content": "Cluster 1: T cells"}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36},
        }
    )

    assert process_openai("genes", "gpt-5.5", "test-key") == ["Cluster 1: T cells"]


@patch("mllmcelltype.providers.openrouter.requests.post")
@patch("mllmcelltype.providers.openrouter.resolve_endpoint_url")
def test_process_openrouter_surfaces_native_cost(mock_resolve_endpoint, mock_post):
    """Test OpenRouter surfaces native USD cost when a usage sink is requested."""
    mock_resolve_endpoint.return_value = "https://openrouter.example/v1"
    mock_post.return_value = _ok_response(
        {
            "choices": [{"message": {"content": "Cluster 1: T cells"}}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "total_tokens": 24,
                "cost": 0.0009,
            },
        }
    )
    sink: dict = {}

    result = process_openrouter("openai/gpt-5.5", "openai/gpt-5.5", "test-key", usage_sink=sink)

    assert result == ["Cluster 1: T cells"]
    assert sink["total_tokens"] == 24
    assert sink["cost"] == 0.0009


def _posted_body(mock_post: MagicMock) -> dict:
    """Decode the JSON body actually sent to requests.post (data= or json=)."""
    kwargs = mock_post.call_args.kwargs
    if "json" in kwargs:
        return kwargs["json"]
    return json.loads(kwargs["data"])


@patch("mllmcelltype.providers.openrouter.requests.post")
@patch("mllmcelltype.providers.openrouter.resolve_endpoint_url")
def test_process_openrouter_omits_deprecated_usage_opt_in(mock_resolve_endpoint, mock_post):
    """Test requests omit the deprecated usage flag with and without a sink."""
    mock_resolve_endpoint.return_value = "https://openrouter.example/v1"
    payload = {"choices": [{"message": {"content": "Cluster 1: T cells"}}]}

    mock_post.return_value = _ok_response(payload)
    process_openrouter("openai/gpt-5.5", "openai/gpt-5.5", "test-key")
    assert "usage" not in _posted_body(mock_post)

    mock_post.return_value = _ok_response(payload)
    process_openrouter("openai/gpt-5.5", "openai/gpt-5.5", "test-key", usage_sink={})
    assert "usage" not in _posted_body(mock_post)


def test_process_gemini_surfaces_usage_through_sink():
    """Test Gemini's usage_metadata is normalized into the caller's usage sink."""
    fake_response = SimpleNamespace(
        text="Cluster 1: T cells",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=5,
            total_token_count=17,
        ),
    )
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=MagicMock(return_value=fake_response))
    )
    fake_modules = _build_fake_google_modules(client=fake_client)
    sink: dict = {}

    with patch.dict("sys.modules", fake_modules, clear=False):
        result = process_gemini("genes", "gemini-3.1-pro-preview", "test-key", usage_sink=sink)

    assert result == ["Cluster 1: T cells"]
    assert sink == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}


# --- Gemini usage extractor: resilience to provider/SDK format drift ---


def test_extract_gemini_usage_ignores_extra_metadata_fields():
    """Unknown usage_metadata attributes are ignored; canonical keys captured."""
    metadata = SimpleNamespace(
        prompt_token_count=40,
        candidates_token_count=10,
        total_token_count=50,
        cached_content_token_count=8,  # future/unknown SDK field
        thoughts_token_count=3,
    )
    usage = extract_gemini_usage(SimpleNamespace(usage_metadata=metadata))
    assert usage == {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50}


def test_extract_gemini_usage_absent_metadata_returns_none():
    """No usage_metadata attribute → None (sink left untouched, no crash)."""
    assert extract_gemini_usage(SimpleNamespace(text="x")) is None


def test_extract_gemini_usage_partial_metadata_reports_none_not_zero():
    """A metadata object missing some counts reports None for those, not a guess."""
    metadata = SimpleNamespace(prompt_token_count=9)  # other counts absent
    usage = extract_gemini_usage(SimpleNamespace(usage_metadata=metadata))
    assert usage == {"prompt_tokens": 9, "completion_tokens": None, "total_tokens": None}


def test_extract_gemini_usage_rejects_invalid_counts():
    """Test Gemini metadata cannot populate malformed token counts."""
    metadata = SimpleNamespace(
        prompt_token_count=-1,
        candidates_token_count=True,
        total_token_count="3",
    )

    assert extract_gemini_usage(SimpleNamespace(usage_metadata=metadata)) == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def test_process_gemini_rejects_invalid_usage_sink_before_client_creation():
    """Test Gemini validates telemetry ownership before creating an SDK client."""
    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=MagicMock()))
    fake_modules = _build_fake_google_modules(client=fake_client)

    with (
        patch.dict("sys.modules", fake_modules, clear=False),
        pytest.raises(ValueError, match="usage_sink must be a mutable mapping"),
    ):
        process_gemini(
            "genes",
            "gemini-3.1-pro-preview",
            "test-key",
            usage_sink=[],  # type: ignore[arg-type]
        )

    fake_modules["google.genai"].Client.assert_not_called()


def test_process_gemini_missing_usage_metadata_leaves_sink_empty():
    """End-to-end: a response without usage_metadata must not crash or touch the sink."""
    fake_response = SimpleNamespace(text="Cluster 1: T cells")  # no usage_metadata
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=MagicMock(return_value=fake_response))
    )
    fake_modules = _build_fake_google_modules(client=fake_client)
    sink: dict = {}

    with patch.dict("sys.modules", fake_modules, clear=False):
        result = process_gemini("genes", "gemini-3.1-pro-preview", "test-key", usage_sink=sink)

    assert result == ["Cluster 1: T cells"]
    assert sink == {}


def test_process_gemini_missing_usage_metadata_clears_stale_sink():
    """A reused sink must not keep previous usage when Gemini omits metadata."""
    fake_response = SimpleNamespace(text="Cluster 1: T cells")  # no usage_metadata
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=MagicMock(return_value=fake_response))
    )
    fake_modules = _build_fake_google_modules(client=fake_client)
    sink: dict = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}

    with patch.dict("sys.modules", fake_modules, clear=False):
        result = process_gemini("genes", "gemini-3.1-pro-preview", "test-key", usage_sink=sink)

    assert result == ["Cluster 1: T cells"]
    assert sink == {}


def test_process_gemini_usage_extraction_failure_is_non_fatal():
    """Test optional Gemini telemetry cannot invalidate a valid model response."""

    class ResponseWithBrokenUsage:
        text = "Cluster 1: T cells"

        @property
        def usage_metadata(self):
            raise RuntimeError("unsupported metadata")

    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=MagicMock(return_value=ResponseWithBrokenUsage()))
    )
    fake_modules = _build_fake_google_modules(client=fake_client)
    sink = {"total_tokens": 99}

    with patch.dict("sys.modules", fake_modules, clear=False):
        result = process_gemini(
            "genes",
            "gemini-3.1-pro-preview",
            "test-key",
            usage_sink=sink,
        )

    assert result == ["Cluster 1: T cells"]
    assert sink == {}


@patch("mllmcelltype.providers.gemini.time.sleep")
def test_process_gemini_retries_exhausted_raises(mock_sleep):
    """Test Gemini raises final error after exhausting retries."""
    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=MagicMock()))
    fake_client.models.generate_content.side_effect = _FakeGoogleAPIError(
        500,
        "persistent failure",
    )
    fake_modules = _build_fake_google_modules(client=fake_client)

    with (
        patch.dict("sys.modules", fake_modules, clear=False),
        pytest.raises(_FakeGoogleAPIError, match="persistent failure"),
    ):
        process_gemini("genes", "gemini-3.1-pro-preview", "test-key")

    assert fake_client.models.generate_content.call_count == 3
    assert [call.args[0] for call in mock_sleep.call_args_list] == [2, 4]


@patch("mllmcelltype.providers.gemini.time.sleep")
def test_process_gemini_failed_call_clears_stale_usage(mock_sleep):
    """Test a failed Gemini call cannot expose token usage from an earlier call."""
    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=MagicMock()))
    fake_client.models.generate_content.side_effect = _FakeGoogleAPIError(
        500,
        "persistent failure",
    )
    fake_modules = _build_fake_google_modules(client=fake_client)
    sink = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}

    with (
        patch.dict("sys.modules", fake_modules, clear=False),
        pytest.raises(_FakeGoogleAPIError, match="persistent failure"),
    ):
        process_gemini(
            "genes",
            "gemini-3.1-pro-preview",
            "test-key",
            usage_sink=sink,
        )

    assert sink == {}


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("broken adapter"),
        _FakeGoogleAPIError(400, "bad request"),
    ],
)
@patch("mllmcelltype.providers.gemini.time.sleep")
def test_process_gemini_non_transient_errors_fail_fast(mock_sleep, error):
    """Test programming and client errors are not amplified by retries."""
    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=MagicMock()))
    fake_client.models.generate_content.side_effect = error
    fake_modules = _build_fake_google_modules(client=fake_client)

    with (
        patch.dict("sys.modules", fake_modules, clear=False),
        pytest.raises(type(error), match=str(error)),
    ):
        process_gemini("genes", "gemini-3.1-pro-preview", "test-key")

    assert fake_client.models.generate_content.call_count == 1
    mock_sleep.assert_not_called()
