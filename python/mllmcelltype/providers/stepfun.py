"""
StepFun provider module for LLMCellType.
"""

import os
import time
from typing import List, Optional
import requests
import json
from ..logger import write_log

def process_stepfun(prompt: str, model: str, api_key: str) -> List[str]:
    """
    Process request using StepFun models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'step-1-8k', 'step-2-16k', 'step-1-flash')
        api_key: StepFun API key

    Returns:
        List[str]: Processed responses, one per cluster
    """
    write_log(f"Starting StepFun API request with model: {model}")

    # Check if API key is provided and not empty
    if not api_key:
        error_msg = "StepFun API key is missing or empty"
        write_log(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    # StepFun API endpoint (OpenAI compatible)
    url = "https://api.stepfun.com/v1/chat/completions"
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
        chunks = [input_lines[i:i+chunk_size] for i in range(0, len(input_lines), chunk_size)]
    else:
        chunks = [input_lines]

    # Process each chunk
    all_results = []
    for i, chunk in enumerate(chunks):
        write_log(f"Processing chunk {i+1} of {cutnum}")

        # Prepare the request body
        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "\n".join(chunk)
                }
            ],
            "temperature": 0.7,
            "max_tokens": 4096
        }

        write_log("Sending API request...")
        # Make the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url=url,
                    headers=headers,
                    data=json.dumps(body)
                )

                # Check for errors
                if response.status_code != 200:
                    error_message = response.json()
                    write_log(f"ERROR: StepFun API request failed: {error_message.get('error', {}).get('message', 'Unknown error')}")

                    # If rate limited, wait and retry
                    if response.status_code == 429:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)
                            write_log(f"Rate limited. Waiting {wait_time} seconds before retrying...")
                            time.sleep(wait_time)
                            continue

                    response.raise_for_status()

                # Parse the response
                content = response.json()
                res = content['choices'][0]['message']['content'].strip().split('\n')
                write_log(f"Got response with {len(res)} lines")
                write_log(f"Raw response from StepFun:\n{res}")

                all_results.extend(res)
                break  # Success, exit retry loop

            except Exception as e:
                write_log(f"Error during API call (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    write_log(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    raise

    write_log("All chunks processed successfully")
    # Clean up results (remove commas at the end of lines)
    return [line.rstrip(',') for line in all_results]
