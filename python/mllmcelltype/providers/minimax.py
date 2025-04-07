"""
MiniMax provider module for LLMCellType.
"""

import os
import time
from typing import List, Optional
import requests
import json
from ..logger import write_log

def process_minimax(prompt: str, model: str, api_key: str, group_id: str = None) -> List[str]:
    """
    Process request using MiniMax models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'minimax-text-02', 'abab6-chat', 'abab5.5-chat')
        api_key: MiniMax API key
        group_id: MiniMax group ID (required for authentication)

    Returns:
        List[str]: Processed responses, one per cluster
    """
    write_log(f"Starting MiniMax API request with model: {model}")

    # Check if API key is provided and not empty
    if not api_key:
        error_msg = "MiniMax API key is missing or empty"
        write_log(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    # Check if group_id is provided
    if not group_id:
        group_id = os.getenv("MINIMAX_GROUP_ID")
        if not group_id:
            error_msg = "MiniMax Group ID is missing or empty"
            write_log(f"ERROR: {error_msg}")
            raise ValueError(error_msg)

    # MiniMax API endpoint - use the same endpoint as in R version
    url = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"
    write_log(f"Using model: {model}")
    write_log(f"API URL: {url}")

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

        # Prepare the request body - use the same format as in R version
        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "name": "user",
                    "content": "\n".join(chunk)
                }
            ]
        }

        write_log("Sending API request...")
        # Make the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Minimax-Group-Id": group_id
        }

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                # Log request details for debugging
                write_log(f"Request URL: {url}")
                write_log(f"Request headers: {headers}")
                write_log(f"Request body: {json.dumps(body)}")

                response = requests.post(
                    url=url,
                    headers=headers,
                    data=json.dumps(body)
                )

                # Log response details
                write_log(f"Response status code: {response.status_code}")
                write_log(f"Response headers: {response.headers}")

                # Check for errors
                if response.status_code != 200:
                    try:
                        error_message = response.json()
                        write_log(f"ERROR: MiniMax API request failed: {error_message}")
                        write_log(f"Error details: {error_message.get('error', {}).get('message', 'Unknown error')}")
                    except Exception as e:
                        write_log(f"ERROR: MiniMax API request failed with status {response.status_code}")
                        write_log(f"Response text: {response.text}")

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

                # Parse response using the same format as in R version
                if 'choices' in content and len(content['choices']) > 0 and 'message' in content['choices'][0] and 'content' in content['choices'][0]['message']:
                    response_content = content['choices'][0]['message']['content']
                    res = response_content.strip().split('\n')
                else:
                    write_log(f"Unexpected response format: {content}")
                    raise ValueError(f"Unexpected response format: {content}")

                write_log(f"Got response with {len(res)} lines")
                write_log(f"Raw response from MiniMax:\n{res}")

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
