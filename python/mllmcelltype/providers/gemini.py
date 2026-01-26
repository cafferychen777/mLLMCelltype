"""Gemini provider module for LLMCellType."""

import time
from typing import Optional

from ..logger import write_log


def process_gemini(
    prompt: str, model: str, api_key: str, base_url: Optional[str] = None
) -> list[str]:
    """Process request using Google Gemini models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'gemini-3-pro', 'gemini-3-flash', 'gemini-2.5-pro')
        api_key: Google API key
        base_url: Optional custom base URL (Note: Gemini uses SDK, base_url may not be applicable)

    Returns:
        List[str]: Processed responses, one per cluster

    Raises:
        ImportError: If google-genai package is not installed

    """
    # Lazy import - only load google-genai when actually using Gemini
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError(
            "Gemini provider requires 'google-genai' package. "
            "Install it with: pip install 'mllmcelltype[gemini]' or pip install google-genai"
        ) from e

    write_log(f"Starting Gemini API request with model: {model}")

    # Warn if base_url is provided (Gemini SDK doesn't support custom URLs)
    if base_url:
        write_log(
            "base_url parameter is ignored for Gemini (SDK doesn't support custom URLs)",
            level="warning",
        )

    # Check if API key is provided and not empty
    if not api_key:
        error_msg = "Google API key is missing or empty"
        write_log(error_msg, level="error")
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
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=4096),
            )

            # Parse the response
            result = response.text.strip().split("\n")
            write_log(f"Got response with {len(result)} lines")
            write_log(f"Raw response from Gemini:\n{result}", level="debug")

            # Clean up results (remove commas at the end of lines)
            return [line.rstrip(",") for line in result]

        except Exception as e:
            write_log(f"Error during API call (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                write_log(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                raise
