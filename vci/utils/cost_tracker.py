"""
VCI Cost Tracker

Tracks token usage and costs per model during benchmark runs.
Uses OpenRouter's pricing data.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from threading import Lock


# OpenRouter pricing per 1M tokens (as of Jan 2026)
# Source: https://openrouter.ai/models
MODEL_PRICING = {
    # OpenAI
    "openai/gpt-5": {"input": 15.00, "output": 60.00},
    "openai/gpt-5.1": {"input": 12.00, "output": 48.00},
    "openai/gpt-5.2": {"input": 10.00, "output": 40.00},
    "openai/o3": {"input": 20.00, "output": 80.00},
    "openai/o3-pro": {"input": 30.00, "output": 120.00},
    "openai/o3-mini": {"input": 3.00, "output": 12.00},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},

    # Anthropic
    "anthropic/claude-opus-4.5": {"input": 15.00, "output": 75.00},
    "anthropic/claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "anthropic/claude-opus-4.1": {"input": 12.00, "output": 60.00},
    "anthropic/claude-opus-4": {"input": 15.00, "output": 75.00},
    "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic/claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},

    # Google
    "google/gemini-3-pro-preview": {"input": 5.00, "output": 20.00},
    "google/gemini-3-flash-preview": {"input": 1.00, "output": 4.00},
    "google/gemini-2.5-pro": {"input": 2.50, "output": 10.00},
    "google/gemini-2.5-flash": {"input": 0.50, "output": 2.00},
    "google/gemini-2.0-flash-001": {"input": 0.40, "output": 1.60},

    # Perplexity
    "perplexity/sonar-pro": {"input": 3.00, "output": 15.00},
    "perplexity/sonar": {"input": 1.00, "output": 1.00},
    "perplexity/sonar-reasoning": {"input": 1.00, "output": 5.00},

    # xAI
    "x-ai/grok-3": {"input": 5.00, "output": 15.00},
    "x-ai/grok-3-mini": {"input": 1.00, "output": 3.00},

    # Meta
    "meta-llama/llama-4-maverick": {"input": 0.80, "output": 0.80},
    "meta-llama/llama-3.3-70b-instruct": {"input": 0.40, "output": 0.40},

    # DeepSeek
    "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19},
    "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
}

# Default pricing for unknown models
DEFAULT_PRICING = {"input": 5.00, "output": 15.00}


@dataclass
class ModelUsage:
    """Usage statistics for a single model."""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    errors: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Calculate cost in USD."""
        pricing = MODEL_PRICING.get(self.model, DEFAULT_PRICING)
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class CostReport:
    """Cost report for a benchmark run."""
    run_id: str
    timestamp: str
    models: Dict[str, ModelUsage] = field(default_factory=dict)
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "models": {k: asdict(v) for k, v in self.models.items()},
            "total_cost_usd": self.total_cost_usd,
            "summary": {
                "total_input_tokens": sum(m.input_tokens for m in self.models.values()),
                "total_output_tokens": sum(m.output_tokens for m in self.models.values()),
                "total_calls": sum(m.calls for m in self.models.values()),
                "total_errors": sum(m.errors for m in self.models.values()),
            }
        }


class CostTracker:
    """Thread-safe cost tracker for benchmark runs."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.usage: Dict[str, ModelUsage] = {}
        self._lock = Lock()

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        error: bool = False
    ):
        """Record token usage for a model call."""
        with self._lock:
            if model not in self.usage:
                self.usage[model] = ModelUsage(model=model)

            self.usage[model].input_tokens += input_tokens
            self.usage[model].output_tokens += output_tokens
            self.usage[model].calls += 1
            if error:
                self.usage[model].errors += 1

    def get_model_cost(self, model: str) -> float:
        """Get cost for a specific model."""
        with self._lock:
            if model in self.usage:
                return self.usage[model].cost_usd
            return 0.0

    def get_total_cost(self) -> float:
        """Get total cost across all models."""
        with self._lock:
            return sum(m.cost_usd for m in self.usage.values())

    def generate_report(self) -> CostReport:
        """Generate a cost report."""
        with self._lock:
            report = CostReport(
                run_id=self.run_id,
                timestamp=datetime.now().isoformat(),
                models=dict(self.usage),
                total_cost_usd=sum(m.cost_usd for m in self.usage.values())
            )
            return report

    def save_report(self, output_dir: Path):
        """Save cost report to JSON file."""
        report = self.generate_report()
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"cost_report_{self.run_id}.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        return report_path

    def print_summary(self):
        """Print a summary of costs."""
        report = self.generate_report()

        print("\n" + "=" * 60)
        print("COST SUMMARY")
        print("=" * 60)
        print(f"Run ID: {report.run_id}")
        print(f"Timestamp: {report.timestamp}")
        print()

        # Sort by cost descending
        sorted_models = sorted(
            report.models.values(),
            key=lambda m: m.cost_usd,
            reverse=True
        )

        print(f"{'Model':<40} {'Calls':>8} {'Tokens':>12} {'Cost':>10}")
        print("-" * 72)

        for m in sorted_models:
            print(f"{m.model:<40} {m.calls:>8} {m.total_tokens:>12,} ${m.cost_usd:>9.4f}")

        print("-" * 72)
        total_tokens = sum(m.total_tokens for m in report.models.values())
        total_calls = sum(m.calls for m in report.models.values())
        print(f"{'TOTAL':<40} {total_calls:>8} {total_tokens:>12,} ${report.total_cost_usd:>9.4f}")
        print("=" * 60)


# Global cost tracker instance (for use across modules)
_global_tracker: Optional[CostTracker] = None
_global_lock = Lock()


def get_global_tracker() -> CostTracker:
    """Get or create the global cost tracker."""
    global _global_tracker
    with _global_lock:
        if _global_tracker is None:
            _global_tracker = CostTracker()
        return _global_tracker


def reset_global_tracker(run_id: Optional[str] = None):
    """Reset the global cost tracker."""
    global _global_tracker
    with _global_lock:
        _global_tracker = CostTracker(run_id)


def record_usage(model: str, input_tokens: int, output_tokens: int, error: bool = False):
    """Record usage to the global tracker."""
    get_global_tracker().record_usage(model, input_tokens, output_tokens, error)
