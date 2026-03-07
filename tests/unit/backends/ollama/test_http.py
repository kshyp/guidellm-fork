"""
Unit tests for OllamaBackend implementation.
"""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from guidellm.backends import Backend
from guidellm.backends.ollama import OllamaBackend
from guidellm.backends.openai.http import OpenAIHTTPBackend


class TestOllamaBackend:
    """Test cases for OllamaBackend."""

    @pytest.fixture(
        params=[
            {"target": "http://localhost:11434"},
            {
                "target": "http://ollama-server:11434",
                "model": "llama3.2",
                "timeout": 30.0,
            },
            {
                "target": "http://localhost:11434",
                "model": "mistral",
                "timeout": 120.0,
                "http2": False,
                "stream": False,
            },
        ]
    )
    def valid_instances(self, request):
        """Fixture providing valid OllamaBackend instances."""
        constructor_args = request.param
        instance = OllamaBackend(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test OllamaBackend inheritance and type relationships."""
        assert issubclass(OllamaBackend, OpenAIHTTPBackend)
        assert issubclass(OllamaBackend, Backend)
        # Check that required methods exist
        assert hasattr(OllamaBackend, "process_startup")
        assert hasattr(OllamaBackend, "process_shutdown")
        assert hasattr(OllamaBackend, "validate")
        assert hasattr(OllamaBackend, "resolve")
        assert hasattr(OllamaBackend, "default_model")
        assert hasattr(OllamaBackend, "available_models")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test OllamaBackend initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, OllamaBackend)
        expected_target = constructor_args["target"].rstrip("/").removesuffix("/v1")
        assert instance.target == expected_target
        assert instance.type_ == "ollama"

        if "model" in constructor_args:
            assert instance.model == constructor_args["model"]
        else:
            assert instance.model == ""

    @pytest.mark.smoke
    def test_default_api_routes(self):
        """Test that OllamaBackend uses Ollama-specific API routes."""
        backend = OllamaBackend(target="http://localhost:11434")

        # Ollama uses /api/tags for model listing
        assert backend.api_routes["/v1/models"] == "api/tags"
        assert backend.api_routes["/health"] == "api/tags"

        # OpenAI-compatible endpoints should remain unchanged
        assert backend.api_routes["/v1/chat/completions"] == "v1/chat/completions"
        assert backend.api_routes["/v1/completions"] == "v1/completions"

    @pytest.mark.smoke
    def test_api_routes_override(self):
        """Test that custom api_routes can override defaults."""
        custom_routes = {"/v1/models": "custom/models"}
        backend = OllamaBackend(
            target="http://localhost:11434", api_routes=custom_routes
        )

        # Custom route should override default
        assert backend.api_routes["/v1/models"] == "custom/models"
        # Other routes should still be defaults
        assert backend.api_routes["/v1/chat/completions"] == "v1/chat/completions"

    @pytest.mark.smoke
    def test_inherits_openai_functionality(self):
        """Test that OllamaBackend inherits OpenAIHTTPBackend functionality."""
        backend = OllamaBackend(target="http://localhost:11434", model="llama3.2")

        # Should have OpenAIHTTPBackend properties
        assert hasattr(backend, "target")
        assert hasattr(backend, "model")
        assert hasattr(backend, "api_key")
        assert hasattr(backend, "stream")
        assert hasattr(backend, "info")

    @pytest.mark.sanity
    def test_default_target_port(self):
        """Test Ollama default port (11434)."""
        backend = OllamaBackend(target="http://localhost:11434")
        assert ":11434" in backend.target or backend.target.endswith("11434")

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_available_models(self, httpx_mock: HTTPXMock):
        """Test available_models with Ollama response format."""
        # Mock Ollama's /api/tags response format
        httpx_mock.add_response(
            url="http://localhost:11434/api/tags",
            json={
                "models": [
                    {"name": "llama3.2:latest", "model": "llama3.2:latest"},
                    {"name": "mistral:7b", "model": "mistral:7b"},
                    {"name": "phi3", "model": "phi3"},
                ]
            },
        )

        backend = OllamaBackend(target="http://localhost:11434")
        await backend.process_startup()

        models = await backend.available_models()

        await backend.process_shutdown()

        # Should include both full names and base names
        assert "llama3.2:latest" in models
        assert "llama3.2" in models
        assert "mistral:7b" in models
        assert "mistral" in models
        assert "phi3" in models

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_available_models_no_tag(self, httpx_mock: HTTPXMock):
        """Test available_models with models that have no tags."""
        httpx_mock.add_response(
            url="http://localhost:11434/api/tags",
            json={
                "models": [
                    {"name": "llama3.2", "model": "llama3.2"},
                ]
            },
        )

        backend = OllamaBackend(target="http://localhost:11434")
        await backend.process_startup()

        models = await backend.available_models()

        await backend.process_shutdown()

        assert "llama3.2" in models
        assert len([m for m in models if m == "llama3.2"]) == 1  # No duplicates

    @pytest.mark.smoke
    def test_backend_registered(self):
        """Test that OllamaBackend is registered in the backend registry."""
        assert Backend.is_registered("ollama")

        # Should be able to create via the registry
        backend = Backend.create("ollama", target="http://localhost:11434")
        assert isinstance(backend, OllamaBackend)

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_backend_not_started_error(self):
        """Test that available_models raises error when backend not started."""
        backend = OllamaBackend(target="http://localhost:11434")

        with pytest.raises(RuntimeError, match="Backend not started up for process"):
            await backend.available_models()

    @pytest.mark.regression
    def test_inheritance_chain(self):
        """Test that OllamaBackend properly inherits from OpenAIHTTPBackend."""
        backend = OllamaBackend(target="http://localhost:11434")

        # Should use OpenAIHTTPBackend's resolve method
        assert backend.resolve.__func__ == OpenAIHTTPBackend.resolve

        # _build_headers is inherited (bound method comparison)
        assert backend._build_headers.__func__ == OpenAIHTTPBackend._build_headers

    @pytest.mark.smoke
    def test_default_model_no_model_set(self):
        """Test default_model returns empty string when no model is set."""
        backend = OllamaBackend(target="http://localhost:11434")
        # When model is not set and backend not started, should return empty
        assert backend.model == ""
