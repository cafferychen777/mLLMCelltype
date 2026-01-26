"""Anthropic provider module for LLMCellType."""

import json
import time
from typing import Optional

import requests

from ..logger import write_log
from ..url_utils import get_default_api_url, validate_base_url


# Model alias mapping: user-friendly names -> official API model IDs
MODEL_ALIASES = {
    # Claude 4.5 series (latest - Nov 2025)
    "claude-opus-4.5": "claude-opus-4-5-20251101",
    "claude-opus-latest": "claude-opus-4-5-20251101",
    "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
    "claude-sonnet-latest": "claude-sonnet-4-5-20250929",
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "claude-haiku-latest": "claude-haiku-4-5-20251001",
    # Claude 4.1 series (Aug 2025)
    "claude-opus-4.1": "claude-opus-4-1-20250805",
    # Claude 4 series (May 2025)
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    # Claude 3.7 series (Feb 2025)
    "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
    # Claude 3.5 series (2024)
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-new": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-old": "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
    # Claude 3 series (2024)
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    # Deprecated models - map to recommended alternatives
    "claude-2": "claude-sonnet-4-5-20250929",
    "claude-2.0": "claude-sonnet-4-5-20250929",
    "claude-2.1": "claude-sonnet-4-5-20250929",
    "claude-3-sonnet": "claude-3-7-sonnet-20250219",
}

# Models that will be retired
DEPRECATED_MODELS = {
    "claude-2": "July 21, 2025",
    "claude-2.0": "July 21, 2025",
    "claude-2.1": "July 21, 2025",
    "claude-3-sonnet": "July 21, 2025",
    "claude-3-opus": "July 21, 2025",
}


def _resolve_model_name(model: str) -> str:
    """Resolve model alias to official API model ID."""
    return MODEL_ALIASES.get(model, model)


def _check_deprecated_model(model: str) -> None:
    """Log warning if model is deprecated."""
    if model in DEPRECATED_MODELS:
        write_log(
            f"Model '{model}' will be retired on {DEPRECATED_MODELS[model]}. "
            "Please migrate to a newer model.",
            level="warning",
        )


def process_anthropic(
    prompt: str, model: str, api_key: str, base_url: Optional[str] = None
) -> list[str]:
    """Process request using Anthropic Claude models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'claude-3-opus', 'claude-sonnet-4-5-20250929')
        api_key: Anthropic API key
        base_url: Optional custom base URL

    Returns:
        List[str]: Processed responses, one per cluster
    """
    write_log(f"Starting Anthropic API request with model: {model}")

    # Validate API key
    if not api_key:
        error_msg = "Anthropic API key is missing or empty"
        write_log(error_msg, level="error")
        raise ValueError(error_msg)

    # Check for deprecated models and resolve aliases
    _check_deprecated_model(model)
    model = _resolve_model_name(model)
    write_log(f"Using model: {model}")

    # Determine API URL
    if base_url:
        if not validate_base_url(base_url):
            raise ValueError(f"Invalid base URL: {base_url}")
        url = base_url
        write_log(f"Using custom base URL: {url}")
    else:
        url = get_default_api_url("anthropic")
        write_log(f"Using default URL: {url}")

    # Prepare request
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
    }

    # Retry configuration
    max_retries = 3
    retry_delay = 2

    write_log("Sending API request...")

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url=url, headers=headers, data=json.dumps(body), timeout=30
            )

            # Handle errors
            if response.status_code != 200:
                try:
                    error_message = response.json()
                    error_detail = error_message.get("error", {}).get(
                        "message", f"model: {model}"
                    )
                    write_log(f"Anthropic API request failed: {error_detail}", level="error")
                except (ValueError, KeyError, json.JSONDecodeError):
                    write_log(
                        f"Anthropic API request failed with status {response.status_code}",
                        level="error",
                    )

                # Retry on rate limit
                if response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    write_log(f"Rate limited. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

            # Parse response
            content = response.json()
            res = content["content"][0]["text"].strip().split("\n")
            write_log(f"Got response with {len(res)} lines")
            write_log(f"Raw response from Anthropic:\n{res}", level="debug")

            # Clean up results
            return [line.rstrip(",") for line in res]

        except Exception as e:
            write_log(f"Error during API call (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                write_log(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                raise
