# .env File Import Feature

## Overview

The .env file import feature allows users to quickly import all their API keys at once by uploading a .env file, instead of manually entering each key one by one.

## How to Use

1. **Prepare your .env file**
   - Create a text file named `.env`
   - Add your API keys in the format: `PROVIDER_API_KEY=your-key-here`
   - See the sample file for reference

2. **Upload the file** (Two methods)

   **Method 1: Click to upload**
   - In the Configuration step, look for the "Import API Keys" section
   - Click "Upload .env File" button
   - Select your .env file

   **Method 2: Drag and drop**
   - Simply drag your .env file from your file explorer
   - Drop it anywhere on the "Import API Keys" section
   - The area will highlight when you hover with the file

3. **Automatic import**
   - The system will automatically:
     - Parse your .env file
     - Import recognized API keys
     - Select providers that have keys imported
     - Show a success/error message

## Supported API Keys

The following environment variable names are recognized:

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic Claude API key
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` - Google Gemini API key
- `GROK_API_KEY` or `XAI_API_KEY` - X.AI Grok API key
- `DEEPSEEK_API_KEY` - DeepSeek API key
- `QWEN_API_KEY` - Alibaba Qwen API key
- `ZHIPU_API_KEY` or `GLM_API_KEY` - Zhipu GLM API key
- `STEPFUN_API_KEY` - StepFun API key
- `MINIMAX_API_KEY` - MiniMax API key
- `OPENROUTER_API_KEY` - OpenRouter API key

## Sample .env File

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google Gemini
GEMINI_API_KEY=your_gemini_api_key

# Add more as needed...
```

## Benefits

1. **Time-saving**: Import all keys at once instead of copy-pasting individually
2. **Error reduction**: Avoid typos when entering keys manually
3. **Reusability**: Use the same .env file across multiple sessions
4. **Organization**: Keep all your API keys in one secure file

## Security Notes

- API keys are only used for the current session
- Keys are not stored on the server
- The .env file is processed client-side in your browser
- Always keep your .env file secure and never share it

## Tips

1. Download the sample .env file as a template
2. Keep your .env file in a secure location
3. Update your .env file when you get new API keys
4. Use comments (lines starting with #) to organize your file
