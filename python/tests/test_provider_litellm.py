"""Tests for the LiteLLM gateway provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mllmcelltype.config import PROVIDER_CONFIGS, get_supported_providers
from mllmcelltype.functions import (
    PROVIDER_FUNCTIONS,
    get_provider,
    validate_provider_model_match,
)
from mllmcelltype.providers.common import NonRetryableProviderError
from mllmcelltype.providers.litellm import (
    PROXY_PREFIX,
    list_litellm_models,
    models_url,
    process_litellm,
    resolve_api_base,
    resolve_gateway_key,
    resolve_sdk_model,
)


class TestRegistration:
    """The provider must be wired into the config and runtime registries."""

    def test_provider_is_configured(self):
        assert "litellm" in get_supported_providers()
        config = PROVIDER_CONFIGS["litellm"]
        assert config.api_key_env_var == "LITELLM_API_KEY"
        # No model_prefixes: LiteLLM routes any vendor's model, so the
        # provider is chosen explicitly rather than inferred from the name.
        assert config.model_prefixes == ()

    def test_provider_is_in_the_runtime_registry(self):
        # The registry validates config/implementation parity and the exact
        # signature at import time, so reaching here at all is meaningful.
        assert "litellm" in PROVIDER_FUNCTIONS
        assert PROVIDER_FUNCTIONS["litellm"] is process_litellm


class TestExplicitSelection:
    """The provider is chosen explicitly, never inferred from the model name."""

    def test_model_name_inference_is_unchanged(self):
        # This provider adds no prefix, so inference must behave exactly as
        # before; it is selected via provider="litellm".
        assert get_provider("gpt-5.5") == "openai"
        assert get_provider("claude-opus-4-7") == "anthropic"
        assert get_provider("anthropic/claude-sonnet-4.6") == "openrouter"

    def test_validation_exempts_the_gateway(self):
        # The one change needed in shared code: without it, provider="litellm"
        # with model="claude-opus-4-7" is rejected as a mismatch, because the
        # name infers to anthropic.
        validate_provider_model_match("litellm", "claude-opus-4-7", "model")
        validate_provider_model_match("litellm", "gemini-2.5-flash", "model")


class TestResolveGatewayKey:
    """A gateway without a master key is a valid, common setup."""

    def test_explicit_key_wins(self, monkeypatch):
        monkeypatch.setenv("LITELLM_API_KEY", "sk-from-env")
        assert resolve_gateway_key("sk-explicit") == "sk-explicit"

    def test_falls_back_to_the_environment(self, monkeypatch):
        monkeypatch.setenv("LITELLM_API_KEY", "sk-from-env")
        assert resolve_gateway_key(None) == "sk-from-env"

    def test_missing_key_is_allowed_not_an_error(self, monkeypatch):
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        # Deliberately NOT ensure_api_key: an unauthenticated gateway is valid.
        assert resolve_gateway_key(None) == ""
        assert resolve_gateway_key("   ") == ""

    def test_whitespace_is_trimmed(self, monkeypatch):
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        assert resolve_gateway_key("  sk-padded  ") == "sk-padded"


class TestModelsUrl:
    def test_derives_from_a_chat_completions_url(self):
        assert models_url("http://localhost:4000/v1/chat/completions") == (
            "http://localhost:4000/v1/models"
        )

    def test_handles_a_path_prefixed_gateway(self):
        assert models_url("https://example.com/litellm/v1/chat/completions") == (
            "https://example.com/litellm/v1/models"
        )

    def test_appends_to_a_bare_base_url(self):
        assert models_url("http://localhost:4000/v1") == "http://localhost:4000/v1/models"
        assert models_url("http://localhost:4000/v1/") == "http://localhost:4000/v1/models"


class TestResolveApiBase:
    """No gateway configured means LiteLLM routes directly."""

    def test_explicit_base_url_wins(self, monkeypatch):
        monkeypatch.setenv("LITELLM_API_BASE", "http://env:4000")
        assert resolve_api_base("http://explicit:4000") == "http://explicit:4000"

    def test_falls_back_to_the_environment(self, monkeypatch):
        monkeypatch.setenv("LITELLM_API_BASE", "http://env:4000")
        assert resolve_api_base(None) == "http://env:4000"

    def test_none_when_no_gateway_is_configured(self, monkeypatch):
        monkeypatch.delenv("LITELLM_API_BASE", raising=False)
        assert resolve_api_base(None) is None
        assert resolve_api_base("  ") is None

    def test_trailing_slash_is_trimmed(self, monkeypatch):
        monkeypatch.delenv("LITELLM_API_BASE", raising=False)
        assert resolve_api_base("http://localhost:4000/") == "http://localhost:4000"


class TestResolveSdkModel:
    """The same model name works with and without a gateway."""

    def test_direct_mode_passes_the_bare_name(self):
        # No gateway: LiteLLM routes to the vendor itself, which is what lets a
        # consensus run mix vendors with no extra infrastructure.
        assert resolve_sdk_model("claude-opus-4-7", None) == "claude-opus-4-7"

    def test_direct_mode_keeps_a_vendor_prefix(self):
        assert resolve_sdk_model("anthropic/claude-opus-4-7", None) == (
            "anthropic/claude-opus-4-7"
        )

    def test_gateway_mode_adds_the_proxy_prefix(self):
        # With a gateway, LiteLLM must forward rather than resolve the vendor.
        assert resolve_sdk_model("claude-opus-4-7", "http://localhost:4000") == (
            "litellm_proxy/claude-opus-4-7"
        )

    def test_an_explicit_proxy_prefix_is_not_doubled(self):
        assert resolve_sdk_model("litellm_proxy/gpt-5.5", "http://localhost:4000") == (
            "litellm_proxy/gpt-5.5"
        )

    def test_rejects_an_empty_model(self):
        with pytest.raises(ValueError, match="model name is required"):
            resolve_sdk_model("", None)

    def test_proxy_prefix_constant(self):
        assert PROXY_PREFIX == "litellm_proxy/"


def _fake_response(content="Cluster 1: T cells", usage=None, cost=None):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice], usage=usage)
    if cost is not None:
        response._hidden_params = {"response_cost": cost}
    return response


@patch("mllmcelltype.providers.litellm._import_litellm")
class TestProcessLiteLLM:
    def test_direct_mode_sends_no_api_base(self, mock_import, monkeypatch):
        monkeypatch.delenv("LITELLM_API_BASE", raising=False)
        litellm = MagicMock()
        litellm.completion.return_value = _fake_response()
        mock_import.return_value = litellm

        result = process_litellm("genes", "claude-opus-4-7", "sk-key")

        assert result == ["Cluster 1: T cells"]
        kwargs = litellm.completion.call_args.kwargs
        assert kwargs["model"] == "claude-opus-4-7"
        assert "api_base" not in kwargs

    def test_gateway_mode_routes_through_the_proxy(self, mock_import, monkeypatch):
        monkeypatch.delenv("LITELLM_API_BASE", raising=False)
        litellm = MagicMock()
        litellm.completion.return_value = _fake_response()
        mock_import.return_value = litellm

        process_litellm(
            "genes", "claude-opus-4-7", "sk-key", base_url="http://localhost:4000"
        )

        kwargs = litellm.completion.call_args.kwargs
        assert kwargs["model"] == "litellm_proxy/claude-opus-4-7"
        assert kwargs["api_base"] == "http://localhost:4000"
        assert kwargs["api_key"] == "sk-key"

    def test_drop_params_defaults_on(self, mock_import, monkeypatch):
        # Without it, one prompt cannot survive a multi-vendor consensus run:
        # providers reject each other's parameters.
        monkeypatch.delenv("LITELLM_API_BASE", raising=False)
        litellm = MagicMock()
        litellm.completion.return_value = _fake_response()
        mock_import.return_value = litellm

        process_litellm("genes", "gpt-5.5", "sk-key")

        assert litellm.completion.call_args.kwargs["drop_params"] is True

    def test_builds_a_single_user_message(self, mock_import, monkeypatch):
        monkeypatch.delenv("LITELLM_API_BASE", raising=False)
        litellm = MagicMock()
        litellm.completion.return_value = _fake_response()
        mock_import.return_value = litellm

        process_litellm("marker genes here", "gpt-5.5", "sk-key")

        assert litellm.completion.call_args.kwargs["messages"] == [
            {"role": "user", "content": "marker genes here"}
        ]

    def test_omits_the_key_when_none_is_set(self, mock_import, monkeypatch):
        # LiteLLM then reads the vendor's own env var, or calls a keyless gateway.
        monkeypatch.delenv("LITELLM_API_BASE", raising=False)
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        litellm = MagicMock()
        litellm.completion.return_value = _fake_response()
        mock_import.return_value = litellm

        process_litellm("genes", "gpt-5.5", "")

        assert "api_key" not in litellm.completion.call_args.kwargs

    def test_honors_normalize_response(self, mock_import, monkeypatch):
        monkeypatch.delenv("LITELLM_API_BASE", raising=False)
        litellm = MagicMock()
        litellm.completion.return_value = _fake_response(content="raw text")
        mock_import.return_value = litellm

        assert (
            process_litellm("genes", "gpt-5.5", "sk-key", normalize_response=False)
            == "raw text"
        )

    def test_captures_usage_and_cost(self, mock_import, monkeypatch):
        monkeypatch.delenv("LITELLM_API_BASE", raising=False)
        litellm = MagicMock()
        litellm.completion.return_value = _fake_response(
            usage={"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
            cost=0.0009,
        )
        mock_import.return_value = litellm
        sink: dict = {}

        process_litellm("genes", "gpt-5.5", "sk-key", usage_sink=sink)

        assert sink["prompt_tokens"] == 20
        assert sink["completion_tokens"] == 4
        assert sink["cost"] == pytest.approx(0.0009)

    def test_survives_a_response_without_usage(self, mock_import, monkeypatch):
        monkeypatch.delenv("LITELLM_API_BASE", raising=False)
        litellm = MagicMock()
        litellm.completion.return_value = _fake_response(usage=None)
        mock_import.return_value = litellm
        sink: dict = {}

        assert process_litellm("genes", "gpt-5.5", "sk-key", usage_sink=sink) == [
            "Cluster 1: T cells"
        ]

    def test_rejects_a_malformed_response(self, mock_import, monkeypatch):
        monkeypatch.delenv("LITELLM_API_BASE", raising=False)
        litellm = MagicMock()
        litellm.completion.return_value = SimpleNamespace(choices=[], usage=None)
        mock_import.return_value = litellm

        with pytest.raises(NonRetryableProviderError, match="Unexpected response format"):
            process_litellm("genes", "gpt-5.5", "sk-key")


class TestListLiteLLMModels:
    @patch("mllmcelltype.providers.litellm.requests.get")
    @patch("mllmcelltype.providers.litellm.resolve_endpoint_url")
    def test_returns_sorted_model_ids(self, mock_resolve, mock_get):
        mock_resolve.return_value = "http://localhost:4000/v1/chat/completions"
        response = MagicMock()
        response.json.return_value = {
            "data": [{"id": "gpt-5.5"}, {"id": "claude-opus-4-7"}, {"id": "gemini-3.1-pro"}]
        }
        mock_get.return_value = response

        assert list_litellm_models(api_key="sk-key") == [
            "claude-opus-4-7",
            "gemini-3.1-pro",
            "gpt-5.5",
        ]
        assert mock_get.call_args.args[0] == "http://localhost:4000/v1/models"

    @patch("mllmcelltype.providers.litellm.requests.get")
    @patch("mllmcelltype.providers.litellm.resolve_endpoint_url")
    def test_omits_the_auth_header_when_keyless(self, mock_resolve, mock_get, monkeypatch):
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        mock_resolve.return_value = "http://localhost:4000/v1/chat/completions"
        response = MagicMock()
        response.json.return_value = {"data": []}
        mock_get.return_value = response

        list_litellm_models()

        assert "Authorization" not in mock_get.call_args.kwargs["headers"]

    @patch("mllmcelltype.providers.litellm.requests.get")
    @patch("mllmcelltype.providers.litellm.resolve_endpoint_url")
    def test_raises_on_an_unexpected_payload(self, mock_resolve, mock_get):
        mock_resolve.return_value = "http://localhost:4000/v1/chat/completions"
        response = MagicMock()
        response.json.return_value = {"unexpected": True}
        mock_get.return_value = response

        with pytest.raises(ValueError, match="Unexpected model list response"):
            list_litellm_models(api_key="sk-key")
