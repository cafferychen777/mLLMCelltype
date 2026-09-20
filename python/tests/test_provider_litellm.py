"""Tests for the LiteLLM gateway provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mllmcelltype.config import PROVIDER_CONFIGS, get_supported_providers
from mllmcelltype.functions import (
    PROVIDER_FUNCTIONS,
    get_provider,
    validate_provider_model_match,
)
from mllmcelltype.providers.litellm import (
    MODEL_PREFIX,
    list_litellm_models,
    models_url,
    process_litellm,
    resolve_gateway_key,
    strip_model_prefix,
)


class TestRegistration:
    """The provider must be wired into the config and runtime registries."""

    def test_provider_is_configured(self):
        assert "litellm" in get_supported_providers()
        config = PROVIDER_CONFIGS["litellm"]
        assert config.api_key_env_var == "LITELLM_API_KEY"
        assert config.model_prefixes == ("litellm/",)

    def test_provider_is_in_the_runtime_registry(self):
        # The registry validates config/implementation parity and the exact
        # signature at import time, so reaching here at all is meaningful.
        assert "litellm" in PROVIDER_FUNCTIONS
        assert PROVIDER_FUNCTIONS["litellm"] is process_litellm


class TestRouting:
    """Model-name routing, including the OpenRouter precedence trap."""

    def test_prefixed_model_routes_to_litellm(self):
        assert get_provider("litellm/gpt-5.5") == "litellm"

    def test_prefixed_model_wins_over_the_openrouter_slash_rule(self):
        # `get_provider` returns "openrouter" for ANY model containing '/'.
        # That rule runs before prefix matching, so without an explicit
        # slash-prefix pass first, every litellm/* model silently routes to
        # OpenRouter and fails with an OpenRouter key.
        assert get_provider("litellm/anthropic/claude-opus-4-7") == "litellm"

    def test_openrouter_still_claims_other_namespaced_models(self):
        assert get_provider("anthropic/claude-sonnet-4.6") == "openrouter"
        assert get_provider("openai/gpt-5.5") == "openrouter"

    def test_bare_models_are_unaffected(self):
        assert get_provider("gpt-5.5") == "openai"
        assert get_provider("claude-opus-4-7") == "anthropic"

    def test_case_insensitive(self):
        assert get_provider("LiteLLM/GPT-5.5") == "litellm"

    def test_validation_exempts_the_gateway(self):
        # The gateway intentionally routes other vendors' models, so a
        # provider/model mismatch is not an error here.
        validate_provider_model_match("litellm", "litellm/claude-opus-4-7", "model")


class TestStripModelPrefix:
    def test_strips_the_routing_prefix(self):
        assert strip_model_prefix("litellm/gpt-5.5") == "gpt-5.5"

    def test_is_case_insensitive(self):
        assert strip_model_prefix("LiteLLM/gpt-5.5") == "gpt-5.5"

    def test_leaves_an_unprefixed_name_alone(self):
        assert strip_model_prefix("gpt-5.5") == "gpt-5.5"

    def test_preserves_a_nested_namespace(self):
        # The gateway may be configured with namespaced aliases of its own.
        assert strip_model_prefix("litellm/anthropic/claude-opus-4-7") == (
            "anthropic/claude-opus-4-7"
        )

    def test_prefix_constant_matches_the_config(self):
        assert PROVIDER_CONFIGS["litellm"].model_prefixes == (MODEL_PREFIX,)


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


@patch("mllmcelltype.providers.litellm.call_openai_compatible_api")
@patch("mllmcelltype.providers.litellm.resolve_endpoint_url")
class TestProcessLiteLLM:
    def test_sends_the_stripped_model_name(self, mock_resolve, mock_call):
        mock_resolve.return_value = "http://localhost:4000/v1/chat/completions"
        mock_call.return_value = ["Cluster 1: T cells"]

        result = process_litellm("genes", "litellm/gpt-5.5", "sk-key")

        assert result == ["Cluster 1: T cells"]
        kwargs = mock_call.call_args.kwargs
        # The gateway knows nothing about the routing prefix.
        assert kwargs["body"]["model"] == "gpt-5.5"
        assert kwargs["provider_name"] == "LiteLLM"
        assert kwargs["api_key"] == "sk-key"

    def test_builds_a_single_user_message(self, mock_resolve, mock_call):
        mock_resolve.return_value = "http://localhost:4000/v1/chat/completions"
        mock_call.return_value = ["Cluster 1: T cells"]

        process_litellm("marker genes here", "litellm/gpt-5.5", "sk-key")

        body = mock_call.call_args.kwargs["body"]
        assert body["messages"] == [{"role": "user", "content": "marker genes here"}]

    def test_works_without_a_key(self, mock_resolve, mock_call, monkeypatch):
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        mock_resolve.return_value = "http://localhost:4000/v1/chat/completions"
        mock_call.return_value = ["Cluster 1: T cells"]

        process_litellm("genes", "litellm/gpt-5.5", "")

        assert mock_call.call_args.kwargs["api_key"] == ""

    def test_forwards_a_custom_base_url(self, mock_resolve, mock_call):
        mock_resolve.return_value = "https://gw.example.com/v1/chat/completions"
        mock_call.return_value = ["Cluster 1: T cells"]

        process_litellm("genes", "litellm/gpt-5.5", "sk-key", base_url="https://gw.example.com")

        assert mock_resolve.call_args.args[2] == "https://gw.example.com"
        assert mock_call.call_args.kwargs["url"] == "https://gw.example.com/v1/chat/completions"

    def test_forwards_the_usage_sink(self, mock_resolve, mock_call):
        mock_resolve.return_value = "http://localhost:4000/v1/chat/completions"
        mock_call.return_value = ["Cluster 1: T cells"]
        sink: dict = {}

        process_litellm("genes", "litellm/gpt-5.5", "sk-key", usage_sink=sink)

        assert mock_call.call_args.kwargs["usage_sink"] is sink

    def test_honors_normalize_response(self, mock_resolve, mock_call):
        mock_resolve.return_value = "http://localhost:4000/v1/chat/completions"
        mock_call.return_value = "raw text"

        process_litellm("genes", "litellm/gpt-5.5", "sk-key", normalize_response=False)

        assert mock_call.call_args.kwargs["normalize_response"] is False

    def test_rejects_an_empty_model_after_the_prefix(self, mock_resolve, mock_call):
        mock_resolve.return_value = "http://localhost:4000/v1/chat/completions"

        with pytest.raises(ValueError, match="No model name left"):
            process_litellm("genes", "litellm/", "sk-key")


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
