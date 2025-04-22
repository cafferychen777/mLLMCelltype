"""Gemini provider module for LLMCellType."""

import time

from google import genai
from google.genai import types

from ..logger import write_log


def process_gemini(prompt: str, model: str, api_key: str) -> list[str]:
    """Process request using Google Gemini models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'gemini-2.0-flash', 'gemini-2.0-pro')
        api_key: Google API key

    Returns:
        List[str]: Processed responses, one per cluster

    """
    write_log(f"Starting Gemini API request with model: {model}")

    # Check if API key is provided and not empty
    if not api_key:
        error_msg = "Google API key is missing or empty"
        write_log(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    # Initialize the client
    client = genai.Client(api_key=api_key)
    write_log(f"Using model: {model}")

    # Set up retry parameters
    max_retries = 3
    retry_delay = 2

    # Try to generate content with retries
    for attempt in range(max_retries):
        try:
            write_log("Sending API request...")

            # Generate content
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7, max_output_tokens=4096
                ),
            )

            # Parse the response
            result = response.text.strip().split("\n")
            write_log(f"Got response with {len(result)} lines")
            write_log(f"Raw response from Gemini:\n{result}")

            # Clean up results (remove commas at the end of lines)
            return [line.rstrip(",") for line in result]

        except Exception as e:
            write_log(
                f"Error during API call (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                write_log(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                raise
    # 如果所有重试都失败，返回空结果
    return ["Unknown"]
