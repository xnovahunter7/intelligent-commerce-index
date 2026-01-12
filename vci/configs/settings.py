"""
VCI Global Settings

Configurable parameters for the evaluation pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Dict

from .providers import WebSearchMode


@dataclass
class ScoringWeights:
    """Weights for the VCI scoring formula by vertical."""
    grounded: float = 0.40
    helpfulness: float = 0.30
    safety: float = 0.15
    completeness: float = 0.15

    def validate(self):
        total = self.grounded + self.helpfulness + self.safety + self.completeness
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


# Default weights per vertical (can be customized)
VERTICAL_WEIGHTS: Dict[str, ScoringWeights] = {
    "fashion": ScoringWeights(
        grounded=0.35,
        helpfulness=0.35,
        safety=0.15,
        completeness=0.15
    ),
    "grocery": ScoringWeights(
        grounded=0.35,
        helpfulness=0.25,
        safety=0.25,  # Higher for dietary/allergen safety
        completeness=0.15
    ),
    "electronics": ScoringWeights(
        grounded=0.45,  # Higher for spec accuracy
        helpfulness=0.25,
        safety=0.15,
        completeness=0.15
    ),
    "travel": ScoringWeights(
        grounded=0.40,
        helpfulness=0.30,
        safety=0.15,
        completeness=0.15
    ),
    "home": ScoringWeights(
        grounded=0.40,
        helpfulness=0.30,
        safety=0.10,
        completeness=0.20  # Higher for delivery/assembly completeness
    ),
}


@dataclass
class PipelineSettings:
    """Settings for the evaluation pipeline."""

    # Grounding verification
    verification_window_hours: int = 2
    max_scrape_retries: int = 3
    scrape_retry_delay_seconds: float = 1.0

    # Parallel processing
    max_workers: int = 100

    # Firecrawl settings
    firecrawl_api_key_env: str = "FIRECRAWL_API_KEY"

    # Grading
    grader_model: str = "gemini-2.0-flash"  # Fast model for grading
    conservative_grading: bool = True  # Unverifiable != wrong

    # Results
    results_dir: str = "results"

    # Web search settings
    # Options: "none", "native", "exa", "auto"
    # - "native": Each provider uses its own search (OpenAI->Bing, Anthropic->their own)
    # - "exa": All models use Exa search (fair/neutral comparison)
    # - "auto": Native for supported providers, Exa for others (OpenRouter default)
    # Default is "native" to reflect real consumer experience
    web_search_mode: str = "native"
    web_search_max_results: int = 3  # Keep low to avoid context overflow


def get_web_search_mode() -> WebSearchMode:
    """Get web search mode from environment or settings."""
    mode_str = os.environ.get("VCI_WEB_SEARCH_MODE", SETTINGS.web_search_mode).lower()
    return WebSearchMode(mode_str)


# Global settings instance
SETTINGS = PipelineSettings()


# Supported verticals
VERTICALS = ["fashion", "grocery", "electronics", "travel", "home"]

# Future verticals (v1.1+)
FUTURE_VERTICALS = ["beauty", "tickets", "luxury", "b2b"]
