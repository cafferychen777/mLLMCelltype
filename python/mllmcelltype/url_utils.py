"""URL utilities for base URL resolution."""

from typing import Optional, Union


def resolve_provider_base_url(provider: str, base_urls: Union[str, dict, None]) -> Optional[str]:
    """Resolve provider-specific base URL.

    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')
        base_urls: User-provided base URLs (string or dict)

    Returns:
        Resolved base URL or None
    """
    if base_urls is None:
        return None

    if isinstance(base_urls, str):
        return base_urls  # Single URL applies to all providers

    if isinstance(base_urls, dict) and provider in base_urls:
        return base_urls[provider]  # Provider-specific URL

    return None


def get_default_api_url(provider: str) -> str:
    """Get default API URL for a provider.

    Args:
        provider: Provider name

    Returns:
        Default API URL for the provider
    """
    default_urls = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
        "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "deepseek": "https://api.deepseek.com/v1/chat/completions",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
        "zhipu": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "grok": "https://api.x.ai/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "stepfun": "https://api.stepfun.com/v1/chat/completions",
        "minimax": "https://api.minimaxi.chat/v1/text/chatcompletion_v2",
    }
    return default_urls.get(provider, "")


def validate_base_url(url: str) -> bool:
    """Validate base URL format.

    Args:
        url: URL to validate

    Returns:
        True if valid, False otherwise
    """
    if not url:
        return False

    # Basic URL format check
    if not (url.startswith("http://") or url.startswith("https://")):
        return False

    return True


def get_working_qwen_endpoint(api_key: str) -> str:
    """Smart endpoint selection for Qwen.

    Args:
        api_key: Qwen API key

    Returns:
        Working endpoint URL
    """

    import requests

    from .logger import write_log

    def test_endpoint_connectivity(endpoint: str, api_key: str, timeout: int = 5) -> bool:
        """Test endpoint connectivity.

        Args:
            endpoint: API endpoint URL
            api_key: API key for authentication
            timeout: Timeout in seconds

        Returns:
            True if endpoint is accessible, False otherwise
        """
        try:
            # Send a simple test request
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            test_body = {
                "model": "qwen-turbo",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
            }

            response = requests.post(endpoint, headers=headers, json=test_body, timeout=timeout)

            # Only 200 or model-related errors (400) indicate the endpoint is reachable
            # 401 and 403 are auth failures, meaning the endpoint doesn't work with this API key
            return response.status_code in [200, 400]

        except Exception:
            return False

    endpoints = [
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",  # International
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",  # Domestic (China)
    ]

    write_log("Testing Qwen endpoint connectivity...", level="debug")

    for i, endpoint in enumerate(endpoints):
        endpoint_type = "international" if i == 0 else "domestic"
        write_log(f"Testing {endpoint_type} endpoint: {endpoint}", level="debug")

        if test_endpoint_connectivity(endpoint, api_key):
            write_log(f"{endpoint_type} endpoint is accessible", level="debug")
            return endpoint
        else:
            write_log(f"{endpoint_type} endpoint is not accessible", level="warning")

    # If none are reachable, return international endpoint as fallback
    write_log("No endpoints accessible, using international endpoint as fallback")
    return endpoints[0]
