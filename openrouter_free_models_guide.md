# Using Free OpenRouter Models with mLLMCelltype

Hi @mhbsiam,

I've investigated the OpenRouter API errors you're encountering. The issue is with the model IDs you're trying to use. Here's what's happening and how to fix it:

## The Problem

1. For free OpenRouter models, you need to add the `:free` suffix to the model ID.
2. The model IDs need to be in the correct format.

Your attempts:
- `microsoft/mai-ds-r1` → Should be `microsoft/mai-ds-r1:free`
- `qwen/qwq-32b-free` → Should be `qwen/qwq-32b:free` (note the colon, not hyphen)

## The Solution

Here's how to use free OpenRouter models correctly:

```python
consensus_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    models=[
        {"provider": "openrouter", "model": "meta-llama/llama-4-maverick:free"},
        {"provider": "openrouter", "model": "nvidia/llama-3.1-nemotron-ultra-253b-v1:free"}
    ],
    api_keys={
        "openrouter": "your-openrouter-api-key"
    },
    max_discussion_rounds=3
)
```

## Recommended Free Models

Here are some high-quality free models with large parameter counts that are currently available on OpenRouter:

### Large Language Models (Sorted by Size)
- `nvidia/llama-3.1-nemotron-ultra-253b-v1:free` - 253B parameters, context: 131K
- `bytedance-research/ui-tars-72b:free` - 72B parameters, context: 32K
- `featherless/qwerky-72b:free` - 72B parameters, context: 32K
- `shisa-ai/shisa-v2-llama3.3-70b:free` - 70B parameters, context: 32K
- `thudm/glm-4-32b:free` - 32B parameters, context: 32K
- `thudm/glm-z1-32b:free` - 32B parameters, context: 32K
- `mistralai/mistral-small-3.1-24b-instruct:free` - 24B parameters, context: 96K
- `agentica-org/deepcoder-14b-preview:free` - 14B parameters, context: 96K

### Models with Extra-Long Context Windows
- `meta-llama/llama-4-scout:free` - 512K context window
- `meta-llama/llama-4-maverick:free` - 256K context window
- `deepseek/deepseek-chat-v3-0324:free` - 163K context window
- `deepseek/deepseek-v3-base:free` - 163K context window
- `tngtech/deepseek-r1t-chimera:free` - 163K context window
- `microsoft/mai-ds-r1:free` - 163K context window

## Checking Available Models

You can check the current list of available models with this code:

```python
import requests
import os

# Get your OpenRouter API key
api_key = "your-openrouter-api-key"

# Get available models
response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

# Print all free models
models = response.json()["data"]
print("Available free OpenRouter models:")
for model in models:
    model_id = model["id"]
    is_free = model.get("pricing", {}).get("prompt") == 0 and model.get("pricing", {}).get("completion") == 0
    if is_free:
        print(f"  - {model_id}")
```

I've updated our documentation to make this clearer for future users. Let me know if you have any other questions!
