#!/bin/bash
# Example: Running GuideLLM benchmark against an Ollama server

# Ensure Ollama is running locally (default port: 11434)
# ollama serve

# Example 1: Simple benchmark with Ollama backend
python -m guidellm benchmark run \
    --target http://localhost:11434 \
    --backend ollama \
    --model llama3.2 \
    --data "synthetic:type=random,decoder=gpt2,min_length=100,max_length=200" \
    --profile sweep \
    --rate 1,5,10

# Example 2: Benchmark with custom backend kwargs
python -m guidellm benchmark run \
    --target http://localhost:11434 \
    --backend ollama \
    --model mistral \
    --data "synthetic:type=random,decoder=gpt2" \
    --profile concurrent \
    --rate 1,2,4 \
    --max-seconds 60 \
    --backend-kwargs '{"timeout": 120, "stream": true}'

# Example 3: Using OpenAI backend with Ollama (Ollama is OpenAI-compatible)
python -m guidellm benchmark run \
    --target http://localhost:11434 \
    --backend openai_http \
    --model llama3.2 \
    --data "synthetic:type=random,decoder=gpt2" \
    --profile sweep
