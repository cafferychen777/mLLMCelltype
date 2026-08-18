"""Low-cost API credential checks for supported model providers."""

import logging
import time
from collections.abc import Callable
from typing import Any

import requests

logger = logging.getLogger(__name__)

try:
    from mllmcelltype.logger import write_log
except ImportError:

    def write_log(message):
        """Fallback logger used when mLLMCelltype is unavailable."""
        logger.info("API validator: %s", message)


try:
    from mllmcelltype.config import get_default_api_url, get_default_model
except ImportError:
    get_default_api_url = None
    get_default_model = None


REQUEST_TIMEOUT = (3.05, 15)
SDK_TIMEOUT_SECONDS = 15


def _provider_base_url(provider: str, fallback: str) -> str:
    """Read the provider endpoint from the annotation package when available."""
    endpoint = get_default_api_url(provider) if get_default_api_url else ""
    suffix = "/chat/completions"
    return (
        endpoint[: -len(suffix)] if endpoint.endswith(suffix) else endpoint or fallback
    )


def _response_error(response: requests.Response, api_key: str) -> str:
    """Extract a bounded provider error without assuming a JSON response."""
    status_messages = {
        401: "Invalid API key",
        403: "Insufficient permissions",
        429: "Rate limit exceeded",
    }
    if response.status_code in status_messages:
        return status_messages[response.status_code]
    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
    else:
        message = error
    cleaned = str(message or f"HTTP {response.status_code}").replace("\n", " ")
    if api_key:
        cleaned = cleaned.replace(api_key, "[redacted]")
    return cleaned[:500]


def _sdk_error(error: Exception, api_key: str) -> str:
    """Classify an SDK error without reflecting credentials or huge responses."""
    message = str(error).replace(api_key, "[redacted]").replace("\n", " ")[:500]
    normalized = message.lower()
    if "authentication" in normalized or "unauthorized" in normalized:
        return "Invalid API key"
    if "permission" in normalized or "forbidden" in normalized:
        return "Insufficient permissions"
    if "rate" in normalized and "limit" in normalized:
        return "Rate limit exceeded"
    if "quota" in normalized:
        return "Quota exceeded"
    if "timeout" in normalized or "timed out" in normalized:
        return "Request timeout"
    return f"API error: {message or type(error).__name__}"


def _post_openai_compatible(
    api_key: str,
    model: str,
    base_url: str,
    *,
    extra_headers: dict[str, str] | None = None,
    max_tokens_parameter: str = "max_tokens",
) -> dict[str, Any]:
    """Send a minimal chat-completions request to a compatible endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **(extra_headers or {}),
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        max_tokens_parameter: 5,
    }
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=body,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        return {"valid": False, "error": "Request timeout"}
    except requests.ConnectionError:
        return {"valid": False, "error": "Connection error"}
    except requests.RequestException as error:
        return {"valid": False, "error": _sdk_error(error, api_key)}

    if response.ok:
        return {"valid": True, "message": "API key is valid"}
    return {"valid": False, "error": _response_error(response, api_key)}


def test_openai_api(api_key: str, model: str) -> dict[str, Any]:
    """Test an OpenAI API key."""
    return _post_openai_compatible(
        api_key,
        model,
        _provider_base_url("openai", "https://api.openai.com/v1"),
        max_tokens_parameter="max_completion_tokens",
    )


def test_anthropic_api(api_key: str, model: str) -> dict[str, Any]:
    """Test an Anthropic API key."""
    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=api_key,
            timeout=SDK_TIMEOUT_SECONDS,
            max_retries=0,
        )
        client.messages.create(
            model=model,
            max_tokens=5,
            messages=[{"role": "user", "content": "Hi"}],
        )
        return {"valid": True, "message": "API key is valid"}
    except ImportError:
        return {"valid": False, "error": "Anthropic SDK not installed"}
    except Exception as error:
        return {"valid": False, "error": _sdk_error(error, api_key)}


def test_gemini_api(api_key: str, model: str) -> dict[str, Any]:
    """Test a Google Gemini API key."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=SDK_TIMEOUT_SECONDS * 1000),
        )
        client.models.generate_content(
            model=model,
            contents="Hi",
            config=types.GenerateContentConfig(max_output_tokens=5),
        )
        return {"valid": True, "message": "API key is valid"}
    except ImportError:
        return {"valid": False, "error": "Google GenAI SDK not installed"}
    except Exception as error:
        return {"valid": False, "error": _sdk_error(error, api_key)}


def test_deepseek_api(api_key: str, model: str) -> dict[str, Any]:
    """Test a DeepSeek API key."""
    return _post_openai_compatible(
        api_key,
        model,
        _provider_base_url("deepseek", "https://api.deepseek.com/v1"),
    )


def test_grok_api(api_key: str, model: str) -> dict[str, Any]:
    """Test an xAI API key."""
    return _post_openai_compatible(
        api_key,
        model,
        _provider_base_url("grok", "https://api.x.ai/v1"),
    )


def test_openrouter_api(api_key: str, model: str) -> dict[str, Any]:
    """Test an OpenRouter API key."""
    return _post_openai_compatible(
        api_key,
        model,
        _provider_base_url("openrouter", "https://openrouter.ai/api/v1"),
        extra_headers={
            "HTTP-Referer": "https://www.mllmcelltype.com",
            "X-Title": "mLLMCelltype",
        },
    )


def test_generic_openai_compatible_api(
    api_key: str, model: str, base_url: str
) -> dict[str, Any]:
    """Test a generic OpenAI-compatible provider."""
    return _post_openai_compatible(api_key, model, base_url)


Validator = Callable[[str, str], dict[str, Any]]


def _compatible_validator(provider: str, fallback_base_url: str) -> Validator:
    def validate(api_key: str, model: str) -> dict[str, Any]:
        return test_generic_openai_compatible_api(
            api_key,
            model,
            _provider_base_url(provider, fallback_base_url),
        )

    return validate


VALIDATOR_FUNCTIONS: dict[str, Validator] = {
    "openai": test_openai_api,
    "anthropic": test_anthropic_api,
    "gemini": test_gemini_api,
    "deepseek": test_deepseek_api,
    "kimi": _compatible_validator("kimi", "https://api.moonshot.cn/v1"),
    "grok": test_grok_api,
    "openrouter": test_openrouter_api,
    "qwen": _compatible_validator(
        "qwen",
        "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    ),
    "zhipu": _compatible_validator("zhipu", "https://api.z.ai/api/paas/v4"),
    "stepfun": _compatible_validator("stepfun", "https://api.stepfun.com/v1"),
    "minimax": _compatible_validator("minimax", "https://api.minimax.io/v1"),
}


def test_provider_api(
    provider: str, api_key: str, model: str | None = None
) -> dict[str, Any]:
    """Test credentials for a supported provider and report elapsed time."""
    provider_name = provider.lower()
    validator = VALIDATOR_FUNCTIONS.get(provider_name)
    if validator is None:
        return {
            "valid": False,
            "error": f"Provider '{provider}' is not supported for testing",
        }

    write_log(f"Testing API key for {provider}")
    started_at = time.monotonic()
    if model is None:
        if not get_default_model:
            return {"valid": False, "error": "Provider model is required"}
        model = get_default_model(provider_name)
        if not model or model == "unknown":
            return {"valid": False, "error": "Provider model is required"}
    result = validator(api_key, model)
    elapsed = time.monotonic() - started_at
    result["response_time"] = round(elapsed, 2)
    write_log(
        f"API test for {provider} completed in {elapsed:.2f}s: "
        f"{'valid' if result.get('valid') else 'invalid'}"
    )
    return result
