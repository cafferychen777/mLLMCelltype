"""DeepSeek provider module for LLMCellType."""

import json
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..logger import write_log


def process_deepseek(prompt: str, model: str, api_key: str) -> list[str]:
    """Process request using DeepSeek models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'deepseek-chat', 'deepseek-coder')
        api_key: DeepSeek API key

    Returns:
        List[str]: Processed responses, one per cluster

    """
    write_log(f"Starting DeepSeek API request with model: {model}")

    # Check if API key is provided and not empty
    if not api_key:
        error_msg = "DeepSeek API key is missing or empty"
        write_log(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    # DeepSeek API endpoint (OpenAI compatible)
    url = "https://api.deepseek.com/v1/chat/completions"
    write_log(f"Using model: {model}")

    # Process all input at once instead of chunks
    input_lines = prompt.split("\n")
    cutnum = 1  # Changed to always use 1 chunk

    write_log(f"Processing {cutnum} chunks of input")

    # Split input into chunks if needed
    if cutnum > 1:
        chunk_size = len(input_lines) // cutnum
        if len(input_lines) % cutnum > 0:
            chunk_size += 1
        chunks = [input_lines[i : i + chunk_size] for i in range(0, len(input_lines), chunk_size)]
    else:
        chunks = [input_lines]

    # Process each chunk
    all_results = []
    for i, chunk in enumerate(chunks):
        write_log(f"Processing chunk {i + 1} of {cutnum}")

        # Prepare the request body
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "\n".join(chunk)}],
            "temperature": 0.7,
            "max_tokens": 4096,
        }

        write_log("Sending API request...")
        # Make the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # Increase retry parameters
        max_retries = 5  # Increased from 3 to 5
        retry_delay = 3  # Increased from 2 to 3
        timeout = 90  # Increased from 60 to 90 seconds

        # Create a session with retry strategy
        session = requests.Session()

        # Configure retry strategy for the session
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )

        # Mount the adapter to the session
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        write_log(
            f"Configured session with {max_retries} retries, {retry_delay}s backoff factor, and {timeout}s timeout"
        )

        for attempt in range(max_retries):
            try:
                write_log(f"Sending request (attempt {attempt + 1}/{max_retries})...")
                response = session.post(url=url, headers=headers, json=body, timeout=timeout)

                # Check for errors
                if response.status_code != 200:
                    error_message = response.json()
                    write_log(
                        f"ERROR: DeepSeek API request failed: {error_message.get('error', {}).get('message', 'Unknown error')}"
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
                write_log(f"Raw response from DeepSeek:\n{res}")

                all_results.extend(res)
                break  # Success, exit retry loop

            except requests.exceptions.Timeout as e:
                # Handle timeout specifically
                write_log(
                    f"Timeout during API call (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    write_log(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    write_log(
                        f"ERROR: All retry attempts failed with timeout. Last error: {str(e)}"
                    )
                    raise

            except requests.exceptions.ConnectionError as e:
                # Handle connection errors specifically
                write_log(
                    f"Connection error during API call (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    write_log(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    write_log(
                        f"ERROR: All retry attempts failed with connection error. Last error: {str(e)}"
                    )
                    raise

            except Exception as e:
                # Handle other exceptions
                write_log(f"Error during API call (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    write_log(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    write_log(f"ERROR: All retry attempts failed. Last error: {str(e)}")
                    raise

    write_log("All chunks processed successfully")
    # Clean up results (remove commas at the end of lines)
    return [line.rstrip(",") for line in all_results]
