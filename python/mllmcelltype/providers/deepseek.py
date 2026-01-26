"""DeepSeek provider module for LLMCellType."""

import time
from typing import Optional

import requests

from ..logger import write_log
from ..url_utils import get_default_api_url, validate_base_url


def process_deepseek(
    prompt: str, model: str, api_key: str, base_url: Optional[str] = None
) -> list[str]:
    """Process request using DeepSeek models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'deepseek-chat', 'deepseek-coder')
        api_key: DeepSeek API key
        base_url: Optional custom base URL

    Returns:
        List[str]: Processed responses, one per cluster

    """
    write_log(f"Starting DeepSeek API request with model: {model}")

    # Check if API key is provided and not empty
    if not api_key:
        error_msg = "DeepSeek API key is missing or empty"
        write_log(error_msg, level="error")
        raise ValueError(error_msg)

    # Use custom URL or default URL
    if base_url:
        if not validate_base_url(base_url):
            raise ValueError(f"Invalid base URL: {base_url}")
        url = base_url
        write_log(f"Using custom base URL: {url}")
    else:
        url = get_default_api_url("deepseek")
        write_log(f"Using default URL: {url}")

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

    # DeepSeek-specific config: longer timeout and more retries for stability
    max_retries = 5
    retry_delay = 3
    timeout = 90

    for attempt in range(max_retries):
        try:
            response = requests.post(url=url, headers=headers, json=body, timeout=timeout)

            # Check for errors
            if response.status_code != 200:
                error_message = response.json()
                write_log(
                    f"DeepSeek API request failed: {error_message.get('error', {}).get('message', 'Unknown error')}",
                    level="error",
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
            write_log(f"Raw response from DeepSeek:\n{res}", level="debug")

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
