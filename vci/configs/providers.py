"""
VCI Provider Configuration

Uses OpenRouter as the unified LLM provider for all models.
OpenRouter provides access to OpenAI, Anthropic, Google, Meta, and other models
through a single API endpoint.

Web Search Modes:
- "native": Use each provider's native search (OpenAI->Bing, Anthropic->their own, etc.)
- "exa": Use Exa search for all models (neutral/fair comparison)
- "none": No web search (baseline)

To enable web search, append ":online" to model ID or use the plugins parameter.
Native search is used for: OpenAI, Anthropic, Perplexity, xAI
Exa search is used for: all other models (Gemini, Llama, Mistral, etc.)
"""

import json
import os
import re
import requests
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class WebSearchMode(Enum):
    """Web search mode for grounding."""
    NONE = "none"        # No web search
    NATIVE = "native"    # Use provider's native search (OpenAI, Anthropic, Perplexity, xAI)
    EXA = "exa"          # Use Exa for all models (neutral comparison)
    AUTO = "auto"        # Native for supported, Exa for others (OpenRouter default)


# Models with native web search support via OpenRouter plugins
# Note: As of Jan 2026, OpenRouter's native web search plugin is inconsistent across providers.
# Google models support native grounding but the OpenRouter "native" engine fails for them.
# Perplexity (sonar) has built-in search that works without the plugin.
# For consistency, we use Exa search for all non-Perplexity models.
# Set this to empty to force Exa for fair comparison.
NATIVE_SEARCH_PROVIDERS = set()  # Empty - use Exa for all (fair comparison)


@dataclass
class ProviderConfig:
    """Configuration for OpenRouter."""
    api_key: str
    base_url: str = OPENROUTER_BASE_URL
    site_url: Optional[str] = None  # For OpenRouter rankings
    site_name: Optional[str] = "VCI Benchmark"
    web_search_mode: WebSearchMode = WebSearchMode.AUTO
    web_search_max_results: int = 5


class OpenRouterProvider:
    """
    OpenRouter provider for making LLM calls with web search.

    OpenRouter provides a unified API for accessing models from:
    - OpenAI (GPT-4o, GPT-4-turbo, etc.)
    - Anthropic (Claude 3.5/4 Sonnet, Opus)
    - Google (Gemini Pro, Flash)
    - Meta (Llama)
    - Perplexity (Sonar)
    - And many more

    Web search can be enabled via:
    1. Append ":online" to model ID (e.g., "openai/gpt-4o:online")
    2. Use plugins parameter with web search config
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": config.site_url or "https://vci-benchmark.com",
            "X-Title": config.site_name or "VCI Benchmark",
        }

    def make_grounded_call(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        web_search: bool = True,
        web_search_mode: Optional[WebSearchMode] = None,
    ) -> dict:
        """
        Make an API call to OpenRouter with optional web search.

        Args:
            prompt: The user prompt
            model: OpenRouter model ID (e.g., "openai/gpt-4o")
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            web_search: Whether to enable web search (default True)
            web_search_mode: Override the default web search mode

        Returns:
            dict with keys:
                - response_text: str
                - grounding_chunks: list[dict]  # URLs from web search
                - grounding_supports: list[dict]  # Citation references
                - metadata: dict
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add web search plugin if enabled
        if web_search:
            search_mode = web_search_mode or self.config.web_search_mode
            plugins = self._build_web_search_plugin(model, search_mode)
            if plugins:
                payload["plugins"] = plugins

        response = requests.post(
            f"{self.config.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

        return self.parse_response(response.json(), model=model)

    def _build_web_search_plugin(
        self,
        model: str,
        search_mode: WebSearchMode
    ) -> Optional[List[Dict]]:
        """Build the web search plugin configuration."""
        if search_mode == WebSearchMode.NONE:
            return None

        # Determine the search engine
        provider = model.split("/")[0] if "/" in model else ""

        if search_mode == WebSearchMode.NATIVE:
            if provider not in NATIVE_SEARCH_PROVIDERS:
                # Native requested but not supported, fall back to Exa
                engine = "exa"
            else:
                engine = "native"
        elif search_mode == WebSearchMode.EXA:
            engine = "exa"
        else:  # AUTO - let OpenRouter decide
            engine = None  # Don't specify, use OpenRouter default

        plugin = {
            "id": "web",
            "max_results": self.config.web_search_max_results,
        }

        if engine:
            plugin["engine"] = engine

        return [plugin]

    def parse_response(self, raw_response: dict, model: str = "") -> dict:
        """Parse OpenRouter response into normalized format."""
        choices = raw_response.get("choices", [])
        if not choices:
            return {
                "response_text": "",
                "grounding_chunks": [],
                "grounding_supports": [],
                "metadata": raw_response
            }

        response_text = choices[0].get("message", {}).get("content", "")

        # Extract URLs from response text as grounding chunks
        grounding_chunks = self._extract_urls(response_text)

        # Extract citation patterns like [1], [2], etc.
        grounding_supports = self._extract_citations(response_text)

        # Track token usage if available
        usage = raw_response.get("usage", {})
        if usage and model:
            try:
                from ..utils.cost_tracker import record_usage
                record_usage(
                    model=model,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0)
                )
            except ImportError:
                pass  # Cost tracking not available

        return {
            "response_text": response_text,
            "grounding_chunks": grounding_chunks,
            "grounding_supports": grounding_supports,
            "metadata": {
                "model": raw_response.get("model"),
                "usage": usage,
                "id": raw_response.get("id"),
            }
        }

    def _extract_urls(self, text: str) -> List[Dict[str, str]]:
        """Extract URLs from response text."""
        url_pattern = r'https?://[^\s<>"\')\]}]+'
        urls = re.findall(url_pattern, text)

        return [{"url": url, "type": "extracted"} for url in urls]

    def _extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """Extract citation references from text."""
        # Match patterns like [1], [2], [Source], etc.
        citation_pattern = r'\[(\d+|[A-Za-z][^\]]*)\]'
        citations = re.findall(citation_pattern, text)

        return [{"reference": cite, "type": "inline"} for cite in citations]


# OpenRouter model IDs
# See: https://openrouter.ai/models
MODEL_REGISTRY = {
    # OpenAI - GPT-5 Family
    "gpt-5": "openai/gpt-5",
    "gpt-5.1": "openai/gpt-5.1",
    "gpt-5.2": "openai/gpt-5.2",
    "o3": "openai/o3",
    "o3-pro": "openai/o3-pro",
    "o3-mini": "openai/o3-mini",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",

    # Anthropic - Claude 4.x Family
    "opus-4.5": "anthropic/claude-opus-4.5",
    "sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "opus-4.1": "anthropic/claude-opus-4.1",
    "opus-4": "anthropic/claude-opus-4",
    "sonnet-4": "anthropic/claude-sonnet-4",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",

    # Google - Gemini Family
    "gemini-3-pro": "google/gemini-3-pro-preview",
    "gemini-3-flash": "google/gemini-3-flash-preview",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",

    # Perplexity (has built-in web search)
    "sonar-pro": "perplexity/sonar-pro",
    "sonar": "perplexity/sonar",
    "sonar-reasoning": "perplexity/sonar-reasoning",

    # xAI
    "grok-3": "x-ai/grok-3",
    "grok-3-mini": "x-ai/grok-3-mini",

    # Meta
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",

    # DeepSeek
    "deepseek-r1": "deepseek/deepseek-r1",
    "deepseek-chat": "deepseek/deepseek-chat",
}

# VCI Benchmark target models (11 flagship models for comparison)
# Ranked by ARC-AGI-2 scores as of Jan 2026
VCI_BENCHMARK_MODELS = [
    # OpenAI (4 models)
    "gpt-5",          # 56.1% - Highest overall
    "gpt-5.1",        # 55.1%
    "o3",             # 52.9%
    "gpt-5.2",        # 51.5%

    # Google (4 models)
    "gemini-3-pro",     # 45.7%
    "gemini-3-flash",   # 36.1%
    "gemini-2.5-flash", # 35.7%
    "gemini-2.5-pro",   # 33.8%

    # Anthropic (3 models)
    "opus-4.5",       # 38.3%
    "sonnet-4.5",     # 35.5%
    "opus-4.1",       # 33.8%
]


def get_openrouter_model_id(model_name: str) -> str:
    """
    Get the OpenRouter model ID for a given model name.

    Accepts either:
    - Short name: "gpt-4o" -> "openai/gpt-4o"
    - Full OpenRouter ID: "openai/gpt-4o" -> "openai/gpt-4o"
    """
    # If it's already a full OpenRouter ID (contains /), return as-is
    if "/" in model_name:
        return model_name

    # Look up in registry
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Use a full OpenRouter ID (e.g., 'openai/gpt-4o') or one of: {list(MODEL_REGISTRY.keys())}"
    )


def load_provider_config(
    web_search_mode: Optional[WebSearchMode] = None,
    web_search_max_results: Optional[int] = None
) -> ProviderConfig:
    """Load OpenRouter configuration from environment variables."""
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing OPENROUTER_API_KEY environment variable. "
            "Get your API key at https://openrouter.ai/keys"
        )

    # Get web search mode from env or default
    if web_search_mode is None:
        mode_str = os.environ.get("VCI_WEB_SEARCH_MODE", "native").lower()
        web_search_mode = WebSearchMode(mode_str)

    # Get max results from env or default
    if web_search_max_results is None:
        web_search_max_results = int(os.environ.get("VCI_WEB_SEARCH_MAX_RESULTS", "3"))

    return ProviderConfig(
        api_key=api_key,
        site_url=os.environ.get("VCI_SITE_URL"),
        site_name=os.environ.get("VCI_SITE_NAME", "VCI Benchmark"),
        web_search_mode=web_search_mode,
        web_search_max_results=web_search_max_results,
    )


def get_provider() -> OpenRouterProvider:
    """Get a configured OpenRouter provider instance."""
    config = load_provider_config()
    return OpenRouterProvider(config)


def list_available_models() -> List[str]:
    """List all available model short names."""
    return list(MODEL_REGISTRY.keys())
