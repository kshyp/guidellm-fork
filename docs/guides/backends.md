# Backends

GuideLLM is designed to work with OpenAI-compatible HTTP servers, enabling seamless integration with a variety of generative AI backends. This compatibility ensures that users can evaluate and optimize their large language model (LLM) deployments efficiently. While the current focus is on OpenAI-compatible servers, we welcome contributions to expand support for other backends, including additional server implementations and Python interfaces.

## Supported Backends

### OpenAI-Compatible HTTP Servers

GuideLLM supports OpenAI-compatible HTTP servers, which provide a standardized API for interacting with LLMs. This includes popular implementations such as [vLLM](https://github.com/vllm-project/vllm), [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference), and [Ollama](https://ollama.com). These servers allow GuideLLM to perform evaluations, benchmarks, and optimizations with minimal setup.

### Ollama

[Ollama](https://ollama.com) is a popular tool for running LLMs locally. Ollama provides OpenAI-compatible endpoints for chat and text completions, making it easy to use with GuideLLM.

GuideLLM includes a dedicated `ollama` backend that:
- Uses Ollama-specific API routes (e.g., `/api/tags` for model listing)
- Handles Ollama's response format for model discovery
- Supports all standard GuideLLM features (streaming, multimodal, etc.)

To use GuideLLM with Ollama:

```bash
# Start Ollama server (if not already running)
ollama serve

# Pull a model (e.g., llama3.2)
ollama pull llama3.2

# Run a benchmark
guidellm benchmark run \
  --target "http://localhost:11434" \
  --backend ollama \
  --model llama3.2 \
  --data "prompt_tokens=256,output_tokens=128" \
  --profile sweep
```

The default Ollama port is `11434`. You can also use the generic `openai_http` backend with Ollama since Ollama supports OpenAI-compatible endpoints.

## Examples for Spinning Up Compatible Servers

### 1. vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-performance OpenAI-compatible server designed for efficient LLM inference. It supports a variety of models and provides a simple interface for deployment.

First ensure you have vLLM installed (`pip install vllm`), and then run the following command to start a vLLM server with a Llama 3.1 8B quantized model:

```bash
vllm serve "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
```

For more information on starting a vLLM server, see the [vLLM Documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

### 2. Text Generation Inference (TGI)

[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) is another OpenAI-compatible server that supports a wide range of models, including those hosted on Hugging Face. TGI is optimized for high-throughput and low-latency inference.

To start a TGI server with a Llama 3.1 8B model using Docker, run the following command:

```bash
docker run --gpus 1 -ti --shm-size 1g --ipc=host --rm -p 8080:80 \
  -e MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct \
  -e NUM_SHARD=1 \
  -e MAX_INPUT_TOKENS=4096 \
  -e MAX_TOTAL_TOKENS=6000 \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  ghcr.io/huggingface/text-generation-inference:2.2.0
```

For more information on starting a TGI server, see the [TGI Documentation](https://huggingface.co/docs/text-generation-inference/index).

### 3. Ollama

[Ollama](https://ollama.com) makes it easy to run LLMs locally with a simple command-line interface and OpenAI-compatible API.

First, [install Ollama](https://ollama.com/download) for your platform. Then start the server and pull a model:

```bash
# Start Ollama server
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.2
```

Now you can run benchmarks with GuideLLM:

```bash
guidellm benchmark run \
  --target "http://localhost:11434" \
  --backend ollama \
  --model llama3.2 \
  --data "prompt_tokens=256,output_tokens=128" \
  --profile sweep
```

For more information, see the [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/README.md).

### 4. llama.cpp

[llama.cpp](https://github.com/ggml-org/llama.cpp) provides lightweight, OpenAI-compatible server through its [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server) tool.

To start a llama.cpp server with the gpt-oss-20b model, you can use the following command:

```bash
llama-server -hf ggml-org/gpt-oss-20b-GGUF --alias gpt-oss-20b --ctx-size 0 --jinja -ub 2048 -b 2048
```

Note that we are providing an alias `gpt-oss-20b` for the model name because `guidellm` is using it to retrieve model metadata in JSON format and such metadata is not included in GGUF model repositories. A simple workaround is to download the metadata files from safetensors repository and place them in a local directory named after the alias:

```bash
huggingface-cli download openai/gpt-oss-20b --include "*.json" --local-dir gpt-oss-20b/
```

Now you can run `guidellm` as usual and it will be able to fetch the model metadata from the local directory.

## API Key Configuration

Some OpenAI-compatible servers require authentication via an API key. This is typically needed when:

- Connecting to OpenAI's API directly
- Using hosted or cloud-based inference services that require authentication
- Connecting to servers that have authentication enabled

Local servers like vLLM typically don't require an API key unless you've explicitly configured authentication.

### Configuring the API Key

To provide an API key when running benchmarks, use the `--backend-kwargs` option:

```bash
guidellm benchmark \
  --target "https://api.openai.com/v1" \
  --backend-kwargs '{"api_key": "sk-..."}' \
  --model "gpt-3.5-turbo" \
  --data "prompt_tokens=256,output_tokens=128"
```

The API key is used to set the `Authorization: Bearer {api_key}` header in HTTP requests to the backend server.

**Note:** For security, avoid hardcoding API keys in scripts. Consider using environment variables or secure credential management tools when passing API keys via `--backend-kwargs`.

## Expanding Backend Support

GuideLLM is an open platform, and we encourage contributions to extend its backend support. Whether it's adding new server implementations, integrating with Python-based backends, or enhancing existing capabilities, your contributions are welcome. For more details on how to contribute, see the [CONTRIBUTING.md](https://github.com/vllm-project/guidellm/blob/main/CONTRIBUTING.md) file.
