"""Qwen provider module for LLMCellType."""

import json
import time
from typing import Optional

import requests

from ..logger import write_log


def process_qwen(
    prompt: str, model: str, api_key: str, base_url: Optional[str] = None
) -> list[str]:
    """Process request using Alibaba Qwen models with smart endpoint selection.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'qwen-plus', 'qwen-max-2025-01-25')
        api_key: DashScope API key
        base_url: Optional custom base URL (overrides smart selection)

    Returns:
        List[str]: Processed responses, one per cluster
    """
    write_log(f"Starting Qwen API request with model: {model}")

    # Check if API key is provided and not empty
    if not api_key:
        error_msg = "DashScope API key is missing or empty"
        write_log(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    # Use custom URL or smart selection
    if base_url:
        from ..url_utils import validate_base_url

        if not validate_base_url(base_url):
            raise ValueError(f"Invalid base URL: {base_url}")
        url = base_url
        write_log(f"Using custom base URL: {url}")
    else:
        from ..url_utils import get_working_qwen_endpoint

        url = get_working_qwen_endpoint(api_key)
        write_log(f"Using smart-selected endpoint: {url}")

    write_log(f"Using model: {model}")

    # Prepare the request body
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4096,
    }

    write_log("Sending API request...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url=url, headers=headers, data=json.dumps(body), timeout=30
            )

            # Check for errors
            if response.status_code != 200:
                error_message = response.json()
                write_log(
                    f"ERROR: Qwen API request failed: {error_message.get('error', {}).get('message', 'Unknown error')}"
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
            res = content["choices"][0]["message"]["content"].strip().split("\n")
            write_log(f"Got response with {len(res)} lines")
            write_log(f"Raw response from Qwen:\n{res}")

            # Clean up results (remove commas at the end of lines)
            return [line.rstrip(",") for line in res]

        except Exception as e:
            write_log(f"Error during API call (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                write_log(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                raise

    # Should not reach here if all retries fail (exception would be raised)
    return []
