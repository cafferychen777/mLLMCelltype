% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/get_provider.R
\name{get_provider}
\alias{get_provider}
\title{Determine provider from model name}
\usage{
get_provider(model)
}
\arguments{
\item{model}{Character string specifying the model name to check}
}
\value{
Character string with the provider name
}
\description{
This function determines the appropriate provider (e.g., OpenAI, Anthropic, Google, OpenRouter) based on the model name.
}
\details{
Supported providers and models include:
\itemize{
\item OpenAI: 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'o1-pro'
\item Anthropic: 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'claude-3-opus'
\item DeepSeek: 'deepseek-chat', 'deepseek-reasoner'
\item Google: 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'
\item Qwen: 'qwen-max-2025-01-25', 'qwen3-72b'
\item Stepfun: 'step-2-mini', 'step-2-16k', 'step-1-8k'
\item Zhipu: 'glm-4-plus', 'glm-3-turbo'
\item MiniMax: 'minimax-text-01'
\item Grok: 'grok-3', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-latest', 'grok-3-mini', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-latest'
\item OpenRouter: Provides access to models from multiple providers through a single API. Format: 'provider/model-name'
\itemize{
\item OpenAI models: 'openai/gpt-4o', 'openai/gpt-4o-mini', 'openai/gpt-4-turbo', 'openai/gpt-4', 'openai/gpt-3.5-turbo'
\item Anthropic models: 'anthropic/claude-3.7-sonnet', 'anthropic/claude-3.5-sonnet',
'anthropic/claude-3.5-haiku', 'anthropic/claude-3-opus'
\item Meta models: 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'meta-llama/llama-2-70b-chat'
\item Google models: 'google/gemini-2.5-pro-preview-03-25', 'google/gemini-1.5-pro-latest', 'google/gemini-1.5-flash'
\item Mistral models: 'mistralai/mistral-large', 'mistralai/mistral-medium', 'mistralai/mistral-small'
\item Other models: 'microsoft/mai-ds-r1', 'perplexity/sonar-small-chat', 'cohere/command-r', 'deepseek/deepseek-chat', 'thudm/glm-z1-32b'
}
}
}
