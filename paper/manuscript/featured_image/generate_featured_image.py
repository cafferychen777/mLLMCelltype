# pip install google-genai Pillow

import mimetypes
import os
from google import genai
from google.genai import types
from PIL import Image


def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_name}")


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-3.1-flash-image-preview"

    prompt = """The following paper has been accepted at Communications Biology. Please generate a Featured Image for the journal's homepage banner (1400x400 pixels, very wide and short, RGB format).

Title: Large Language Model Consensus Substantially Improves the Cell Type Annotation Accuracy for scRNA-seq Data

Abstract: The rapid expansion of single-cell RNA sequencing (scRNA-seq) has made accurate cell type annotation a critical bottleneck for biological discovery. Existing computational methods are often limited by reference data dependency, while emerging single Large Language Model (LLM) approaches are susceptible to model-specific biases and provide insufficient uncertainty quantification. To address these limitations, we introduce mLLMCelltype, a framework that harnesses collective intelligence—the emergent problem-solving capacity arising when multiple independent agents interact through structured deliberation to produce solutions exceeding individual capabilities—of multiple LLMs through an iterative deliberation process. Across 49 diverse datasets, our framework achieves a mean accuracy of 77.2%, a 15.7-percentage-point improvement over the best-performing single-LLM baseline (61.5%). The consensus mechanism demonstrates high robustness to noisy input and generalizes to datasets released after the LLMs' training. By providing transparent reasoning chains and robust consensus-based confidence metrics, mLLMCelltype minimizes manual annotation effort and enables reliable interpretation of complex cellular landscapes. The framework is available as an open-source package and an accessible web server.

Keywords: Cell type annotation, Collective intelligence, Large language models, Multi-LLM consensus, scRNA-seq, Uncertainty quantification

Key Results:
- The mLLMCelltype framework integrates multiple LLMs (GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro, etc.) to annotate cell types from scRNA-seq data through a structured, multi-stage deliberation process.
- Annotation accuracy steadily increases with more LLMs, rising from 59.3% with one LLM to 75.8% with six. For hard clusters, the six-LLM consensus improved accuracy by 38%.
- Iterative deliberation actively reduces hallucinations across rounds, and consensus proportion strongly correlates with annotation reliability.
- Across 49 diverse datasets, mLLMCelltype achieved 77.2% mean accuracy, a 15.7-percentage-point improvement over the best single-LLM baseline.
- The framework is robust to noisy marker gene input and generalizes to unseen data released after LLM training cutoffs.

Please generate a visually striking, scientifically elegant banner image that captures the essence of this work.
"""

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="MINIMAL",
        ),
        image_config=types.ImageConfig(
            image_size="1K",
        ),
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
    )

    file_index = 0
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.parts is None:
            continue
        if chunk.parts[0].inline_data and chunk.parts[0].inline_data.data:
            file_name = f"featured_image_raw_{file_index}"
            file_index += 1
            inline_data = chunk.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            raw_path = f"{file_name}{file_extension}"
            save_binary_file(raw_path, data_buffer)

            # Resize to 1400x400 for Communications Biology
            img = Image.open(raw_path)
            print(f"Original size: {img.size}")
            img_resized = img.resize((1400, 400), Image.LANCZOS)
            img_resized.save("featured_image_1400x400.png", "PNG")
            print("Resized image saved to: featured_image_1400x400.png")
        else:
            if text := chunk.text:
                print(text)


if __name__ == "__main__":
    generate()
