"""
Ollama backend implementation for GuideLLM.

Provides HTTP-based backend for Ollama servers. Ollama supports OpenAI-compatible
API endpoints for chat completions (/v1/chat/completions), but has its own
endpoint for model listing (/api/tags).

This backend extends the OpenAI HTTP backend with Ollama-specific configurations
and model listing handling.

Example:
::
    backend = OllamaBackend(
        target="http://localhost:11434",
        model="llama3.2"
    )

    await backend.process_startup()
    async for response, request_info in backend.resolve(request, info):
        process_response(response)
    await backend.process_shutdown()
"""

from __future__ import annotations

from typing import Any

from guidellm.backends.backend import Backend
from guidellm.backends.openai.http import OpenAIHTTPBackend

__all__ = [
    "OllamaBackend",
]

# Ollama API paths - uses OpenAI-compatible endpoints for chat/completions
# but has its own endpoint for model listing
OLLAMA_API_PATHS = {
    "/health": "api/tags",  # Ollama doesn't have a dedicated health endpoint
    "/v1/models": "api/tags",  # Ollama uses /api/tags for model listing
    "/v1/completions": "v1/completions",
    "/v1/chat/completions": "v1/chat/completions",
    "/v1/audio/transcriptions": "v1/audio/transcriptions",
    "/v1/audio/translations": "v1/audio/translations",
}


@Backend.register("ollama")
class OllamaBackend(OpenAIHTTPBackend):
    """
    HTTP backend for Ollama servers.

    Ollama supports OpenAI-compatible API endpoints for chat and text completions,
    but uses its own endpoint (/api/tags) for model listing. This backend extends
    OpenAIHTTPBackend with Ollama-specific API routes and model listing handling.

    By default, Ollama runs on port 11434. No API key is required for local
    instances.

    Example:
    ::
        backend = OllamaBackend(
            target="http://localhost:11434",
            model="llama3.2"
        )

        await backend.process_startup()
        async for response, request_info in backend.resolve(request, info):
            process_response(response)
        await backend.process_shutdown()
    """

    def __init__(
        self,
        target: str,
        model: str = "",
        request_format: str | None = None,
        api_key: str | None = None,
        api_routes: dict[str, str] | None = None,
        request_handlers: dict[str, Any] | None = None,
        timeout: float | None = None,
        timeout_connect: float | None = None,
        http2: bool = True,
        follow_redirects: bool = True,
        verify: bool = False,
        validate_backend: bool | str | dict[str, Any] = True,
        stream: bool = True,
        extras: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
    ):
        """
        Initialize Ollama backend with server configuration.

        :param target: Base URL of the Ollama server (e.g., http://localhost:11434)
        :param model: Model identifier for generation requests (e.g., llama3.2, mistral)
        :param request_format: Format to use for requests (default: /v1/chat/completions)
        :param api_key: API key for authentication (typically not needed for local Ollama)
        :param api_routes: Custom API endpoint routes mapping
        :param request_handlers: Custom response handlers for different request types
        :param timeout: Request timeout in seconds
        :param timeout_connect: Connection timeout in seconds
        :param http2: Enable HTTP/2 protocol support
        :param follow_redirects: Follow HTTP redirects automatically
        :param verify: Enable SSL certificate verification
        :param validate_backend: Backend validation configuration
        :param stream: Enable streaming responses
        :param extras: Extra arguments to pass to requests
        :param max_tokens: Maximum tokens to generate
        :param max_completion_tokens: Alias for max_tokens for OpenAI compatibility
        """
        # Use Ollama-specific API routes by default, but allow override
        merged_api_routes = {**OLLAMA_API_PATHS, **(api_routes or {})}

        super().__init__(
            target=target,
            model=model,
            request_format=request_format,
            api_key=api_key,
            api_routes=merged_api_routes,
            request_handlers=request_handlers,
            timeout=timeout,
            timeout_connect=timeout_connect,
            http2=http2,
            follow_redirects=follow_redirects,
            verify=verify,
            validate_backend=validate_backend,
            stream=stream,
            extras=extras,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
        )

        # Override the type to identify as ollama backend
        self.type_ = "ollama"  # type: ignore[misc]

    async def available_models(self) -> list[str]:
        """
        Get available models from the Ollama server.

        Ollama's /api/tags endpoint returns a different format than OpenAI's
        /v1/models endpoint. This method handles the Ollama-specific format.

        :return: List of model identifiers
        :raises httpx.HTTPError: If models endpoint returns an error
        :raises RuntimeError: If backend is not initialized
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")

        # Use the api_routes mapping for the models endpoint
        models_path = self.api_routes.get("/v1/models", "api/tags")
        target = f"{self.target}/{models_path}"
        response = await self._async_client.get(target, headers=self._build_headers())
        response.raise_for_status()

        data = response.json()

        # Ollama's /api/tags returns {"models": [{"name": "model:tag", ...}, ...]}
        # Extract just the model name (without tag) for consistency
        models = data.get("models", [])
        model_names = []
        for model in models:
            name = model.get("name", "")
            # Ollama model names often include tags like "llama3.2:latest"
            # We return the full name, but also provide the base name
            if name:
                model_names.append(name)
                # Also add base name without tag if there's a tag
                if ":" in name:
                    base_name = name.split(":")[0]
                    if base_name not in model_names:
                        model_names.append(base_name)

        return model_names
