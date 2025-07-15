"""Anthropic provider module for LLMCellType."""

import json
import time
from typing import Optional

import requests

from ..logger import write_log


def process_anthropic(
    prompt: str, model: str, api_key: str, base_url: Optional[str] = None
) -> list[str]:
    """Process request using Anthropic Claude models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'claude-3-opus', 'claude-3-sonnet')
        api_key: Anthropic API key
        base_url: Optional custom base URL

    Returns:
        List[str]: Processed responses, one per cluster

    """
    write_log(f"Starting Anthropic API request with model: {model}")

    # Check if API key is provided and not empty
    if not api_key:
        error_msg = "Anthropic API key is missing or empty"
        write_log(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    # Check for deprecated models that will be retired on July 21, 2025
    deprecated_models = {
        "claude-2": "July 21, 2025",
        "claude-2.0": "July 21, 2025",
        "claude-2.1": "July 21, 2025",
        "claude-3-sonnet": "July 21, 2025",  # Non-versioned
        "claude-3-opus": "July 21, 2025",  # Non-versioned
    }

    if model in deprecated_models:
        write_log(
            f"WARNING: Model '{model}' will be retired on {deprecated_models[model]}. Please migrate to a newer model."
        )
        write_log("Recommended migrations:")
        if model.startswith("claude-2"):
            write_log("  - Use 'claude-sonnet-4-20250514' or 'claude-3-5-sonnet-20241022'")
        elif model == "claude-3-sonnet":
            write_log("  - Use 'claude-sonnet-4-20250514' or 'claude-3-7-sonnet-20250219'")
        elif model == "claude-3-opus":
            write_log("  - Use 'claude-opus-4-20250514' or 'claude-3-opus-20240229'")

    # Handle old model names and map to the latest versions
    model_mapping = {
        # Claude 4 series
        "claude-opus-4-20250514": "claude-opus-4-20250514",
        "claude-opus-4": "claude-opus-4-20250514",
        "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        # Claude 3.7 series (deprecated - use Claude 4 instead)
        "claude-3-7-sonnet-20250219": "claude-sonnet-4-20250514",
        "claude-3-7-sonnet": "claude-sonnet-4-20250514",
        # Claude 3.5 series
        "claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-new": "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-old": "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",  # Default to new version
        "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
        "claude-3-5-haiku": "claude-3-5-haiku-20241022",
        "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
        # Claude 3 series
        "claude-3-opus-20240229": "claude-3-opus-20240229",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-haiku-20240307": "claude-3-haiku-20240307",
        "claude-3-haiku": "claude-3-haiku-20240307",
        # Deprecated models - mapping to recommended alternatives (will be retired July 21, 2025)
        "claude-2": "claude-3-5-sonnet-20241022",
        "claude-2.0": "claude-3-5-sonnet-20241022",
        "claude-2.1": "claude-3-5-sonnet-20241022",
        "claude-3-sonnet": "claude-3-7-sonnet-20250219",  # Non-versioned, map to versioned
    }

    # Map the model name to the latest version if necessary
    if model in model_mapping:
        model = model_mapping[model]

    write_log(f"Using model: {model}")

    # If base_url is provided, use direct API method
    if base_url:
        write_log("Using direct API method due to custom base_url")
        return process_anthropic_direct(prompt, model, api_key, base_url)

    try:
        # Try to import Anthropic client
        try:
            import anthropic
        except ImportError as err:
            raise ImportError(
                "Anthropic Python SDK not installed. Please install with 'pip install anthropic'."
            ) from err

        # Create client
        client = anthropic.Anthropic(api_key=api_key)

        # Send the message
        write_log("Sending API request to Anthropic...")
        response = client.messages.create(
            model=model, max_tokens=4000, messages=[{"role": "user", "content": prompt}]
        )

        # Get response content
        content = response.content[0].text
        lines = content.strip().split("\n")

        write_log(f"Got response with {len(lines)} lines")
        write_log(f"Raw response from Anthropic:\n{lines}")

        # Count the number of expected lines (clusters)
        input_lines = prompt.split("\n")
        expected_lines = max(0, len(input_lines) - 3)  # -3 for header lines

        # If we got fewer lines than expected, pad with "Unknown"
        if len(lines) < expected_lines:
            write_log(
                f"Warning: Got {len(lines)} lines but expected {expected_lines}. Padding with 'Unknown'."
            )
            lines.extend(["Unknown"] * (expected_lines - len(lines)))

        # If we got more lines than expected, truncate
        if len(lines) > expected_lines:
            write_log(f"Warning: Got {len(lines)} lines but expected {expected_lines}. Truncating.")
            lines = lines[:expected_lines]

        # Clean up response
        return [line.rstrip(",") for line in lines]

    except (
        requests.RequestException,
        ValueError,
        ImportError,
        AttributeError,
        json.JSONDecodeError,
    ) as e:
        write_log(f"Error during Anthropic API call: {str(e)}", level="error")

        # Try alternative method with direct REST API if SDK fails
        return process_anthropic_direct(prompt, model, api_key, base_url)


def process_anthropic_direct(
    prompt: str, model: str, api_key: str, base_url: Optional[str] = None
) -> list[str]:
    """Fallback method using direct API calls if the SDK fails"""

    write_log("Falling back to direct API calls for Anthropic")

    # 使用自定义URL或默认URL
    if base_url:
        from ..url_utils import validate_base_url

        if not validate_base_url(base_url):
            raise ValueError(f"Invalid base URL: {base_url}")
        url = base_url
        write_log(f"Using custom base URL: {url}")
    else:
        from ..url_utils import get_default_api_url

        url = get_default_api_url("anthropic")
        write_log(f"Using default URL: {url}")

    # Process all input at once
    input_lines = prompt.split("\n")
    expected_lines = max(0, len(input_lines) - 3)  # -3 for header lines

    # Prepare the request body
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
    }

    write_log("Sending direct API request...")
    # Make the API request
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.post(url=url, headers=headers, data=json.dumps(body), timeout=30)

            # Check for errors
            if response.status_code != 200:
                try:
                    error_message = response.json()
                    error_detail = error_message.get("error", {}).get("message", f"model: {model}")
                    write_log(f"ERROR: Anthropic API request failed: {error_detail}")
                except (ValueError, KeyError, json.JSONDecodeError):
                    write_log(
                        f"ERROR: Anthropic API request failed with status {response.status_code}"
                    )

                # If rate limited, wait and retry
                if response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    write_log(f"Rate limited. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

            # Parse the response
            content = response.json()
            res = content["content"][0]["text"].strip().split("\n")
            write_log(f"Got response with {len(res)} lines")

            # If we got fewer lines than expected, pad with "Unknown"
            if len(res) < expected_lines:
                write_log(
                    f"Warning: Got {len(res)} lines but expected {expected_lines}. Padding with 'Unknown'."
                )
                res.extend(["Unknown"] * (expected_lines - len(res)))

            # If we got more lines than expected, truncate
            if len(res) > expected_lines:
                write_log(
                    f"Warning: Got {len(res)} lines but expected {expected_lines}. Truncating."
                )
                res = res[:expected_lines]

            # Clean up results (remove commas at the end of lines)
            return [line.rstrip(",") for line in res]

        except (
            requests.RequestException,
            ValueError,
            json.JSONDecodeError,
            KeyError,
        ) as e:
            write_log(
                f"Error during direct API call (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                write_log(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                # If all attempts failed, return empty results
                write_log("All API attempts failed. Returning empty results.", level="error")
                return ["Unknown"] * expected_lines
    # 如果所有重试都失败，返回空结果
    return ["Unknown"] * expected_lines
