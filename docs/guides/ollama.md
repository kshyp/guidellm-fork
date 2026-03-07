# Ollama Backend Guide

This guide covers using GuideLLM with [Ollama](https://ollama.com), a popular tool for running large language models locally.

## Overview

Ollama provides OpenAI-compatible API endpoints, making it easy to benchmark local LLMs with GuideLLM. GuideLLM includes a dedicated `ollama` backend that handles Ollama-specific features like the `/api/tags` model listing endpoint.

## Prerequisites

1. Install Ollama: https://ollama.com/download
2. Start the Ollama server
3. Pull the model(s) you want to benchmark

```bash
# Start Ollama server
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.2
```

## Quick Start

Run a simple benchmark against Ollama:

```bash
guidellm benchmark run \
  --target http://localhost:11434 \
  --backend ollama \
  --model llama3.2 \
  --processor gpt2 \
  --data "prompt_tokens=128,output_tokens=64" \
  --profile sweep \
  --rate 5 \
  --max-seconds 60
```

## Configuration

### Backend Selection

Use `--backend ollama` to select the Ollama backend:

```bash
guidellm benchmark run \
  --target http://localhost:11434 \
  --backend ollama \
  ...
```

You can also use `--backend openai_http` since Ollama supports OpenAI-compatible endpoints, but the `ollama` backend provides better model discovery.

### Ollama Environment Variables

For optimal performance, configure Ollama with these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_NUM_PARALLEL` | 1 | Number of simultaneous requests to process |
| `OLLAMA_MAX_LOADED_MODELS` | 1 | Maximum models to keep loaded in memory |
| `OLLAMA_NUM_THREAD` | varies | CPU threads per request |
| `OLLAMA_KEEP_ALIVE` | 5m | How long to keep models loaded |

Example for high-throughput benchmarking:

```bash
OLLAMA_NUM_PARALLEL=16 \
OLLAMA_MAX_LOADED_MODELS=2 \
OLLAMA_NUM_THREAD=8 \
ollama serve
```

## Benchmarking Patterns

### 1. Synchronous (Sequential) Testing

Best for measuring baseline latency:

```bash
guidellm benchmark run \
  --target http://localhost:11434 \
  --backend ollama \
  --model llama3.2 \
  --processor gpt2 \
  --data "prompt_tokens=512,output_tokens=256" \
  --profile synchronous \
  --max-requests 10
```

### 2. Concurrent Load Testing

Test with fixed concurrency:

```bash
guidellm benchmark run \
  --target http://localhost:11434 \
  --backend ollama \
  --model llama3.2 \
  --processor gpt2 \
  --data "prompt_tokens=128,output_tokens=64" \
  --profile concurrent \
  --rate 8 \
  --max-seconds 120
```

### 3. Throughput Testing

Automatically find maximum throughput:

```bash
guidellm benchmark run \
  --target http://localhost:11434 \
  --backend ollama \
  --model llama3.2 \
  --processor gpt2 \
  --data "prompt_tokens=128,output_tokens=64" \
  --profile throughput \
  --max-seconds 120
```

### 4. Rate Sweep

Test multiple load levels:

```bash
guidellm benchmark run \
  --target http://localhost:11434 \
  --backend ollama \
  --model llama3.2 \
  --processor gpt2 \
  --data "prompt_tokens=128,output_tokens=64" \
  --profile sweep \
  --rate 10 \
  --max-seconds 300
```

## Data Options

### Synthetic Data

Generate synthetic prompts with specific token counts:

```bash
--data "prompt_tokens=256,output_tokens=128"
--data "prompt_tokens=512,output_tokens=256,decoder=gpt2"
```

### HuggingFace Datasets

Use real datasets:

```bash
guidellm benchmark run \
  --target http://localhost:11434 \
  --backend ollama \
  --model llama3.2 \
  --data "abisee/cnn_dailymail" \
  --data-args '{"name": "3.0.0"}' \
  --data-column-mapper '{"text_column":"article"}' \
  --profile concurrent \
  --rate 4 \
  --max-seconds 300
```

## Troubleshooting

### Model Not Found

If you get a model error, ensure the model is pulled:

```bash
ollama list
ollama pull llama3.2
```

### Connection Refused

Check Ollama is running and accessible:

```bash
curl http://localhost:11434/api/tags
```

For remote Ollama, ensure `OLLAMA_HOST` is set:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### Low Throughput

1. Check `OLLAMA_NUM_PARALLEL` is set high enough
2. Verify GPU is being used: `ollama ps`
3. Try different `--profile` options
4. Adjust `OLLAMA_NUM_THREAD` if CPU-bound

## Examples

See `examples/ollama_example.sh` for complete working examples.

## References

- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/README.md)
- [Ollama API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [GuideLLM Backends Guide](./backends.md)
