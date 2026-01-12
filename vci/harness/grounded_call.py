"""
VCI Grounded Call Module

Stage 1 of Pipeline: Makes API calls to AI models via OpenRouter.
Captures responses with citation metadata for downstream verification.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..configs.providers import get_provider, get_openrouter_model_id


@dataclass
class GroundedResponse:
    """Normalized response from a grounded API call."""
    response_text: str
    grounding_chunks: List[Dict[str, Any]]  # Source citations from API
    grounding_supports: List[Dict[str, Any]]  # Citation references in text
    query: str
    model: str
    task_id: str
    criteria: List[Dict[str, Any]]
    timestamp: float
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path):
        """Save response to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "GroundedResponse":
        """Load response from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class TestCase:
    """A test case for evaluation."""
    task_id: str
    prompt: str
    specified_prompt: str  # Enhanced prompt asking for explicit details
    vertical: str
    criteria: List[Dict[str, Any]]

    @classmethod
    def from_dict(cls, data: dict) -> "TestCase":
        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            specified_prompt=data.get("specified_prompt", data["prompt"]),
            vertical=data["vertical"],
            criteria=data.get("criteria", [])
        )

    def to_dict(self) -> dict:
        return asdict(self)


# System prompt for grounded shopping assistance
GROUNDING_SYSTEM_PROMPT = """You are a helpful shopping assistant. When recommending products:

1. ALWAYS include direct links to purchase pages
2. ALWAYS state exact current prices
3. ALWAYS verify availability before recommending
4. Include specific product details (specs, dimensions, materials)
5. Mention return policies and warranties when relevant
6. If you're unsure about current availability or pricing, say so

Be specific and accurate. Users rely on your recommendations to make purchasing decisions."""


def build_specified_prompt(prompt: str, vertical: str) -> str:
    """
    Build a specified prompt that asks the model to be explicit about
    verifiable details (prices, specs, availability, etc.).
    """
    vertical_instructions = {
        "fashion": """
Please provide for each recommendation:
- Exact price (including shipping if applicable)
- Available sizes
- Material/composition
- Return policy details
- Direct link to purchase page""",

        "grocery": """
Please provide for each recommendation:
- Exact price and unit pricing
- Availability for delivery to the specified location
- Delivery window options
- Any relevant allergen information
- Direct link to purchase or store page""",

        "electronics": """
Please provide for each recommendation:
- Exact specifications (RAM, storage, screen size, processor, etc.)
- Current price
- Model name and year
- Warranty information
- Compatibility details if relevant
- Direct link to purchase from authorized seller""",

        "travel": """
Please provide for each recommendation:
- Exact dates and times
- Total cost including ALL fees and taxes
- Cancellation policy
- Direct booking links
- Any entry requirements or restrictions""",

        "home": """
Please provide for each recommendation:
- Exact dimensions (width, height, depth)
- Current price including delivery cost
- Delivery timeline
- Assembly requirements
- Return policy
- Direct link to purchase page"""
    }

    instruction = vertical_instructions.get(vertical, "")
    if instruction:
        return f"{prompt}\n{instruction}"
    return prompt


def make_grounded_call(
    test_case: TestCase,
    model: str,
    output_path: Optional[Path] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> GroundedResponse:
    """
    Make a grounded API call via OpenRouter with retry logic.

    Args:
        test_case: The test case containing prompt and criteria
        model: Model name (e.g., "gpt-4o", "claude-3.5-sonnet", or full OpenRouter ID)
        output_path: Optional path to save the response
        max_retries: Maximum number of retry attempts for transient failures
        retry_delay: Base delay between retries (exponential backoff applied)

    Returns:
        GroundedResponse with normalized response data

    Raises:
        Exception: If all retries are exhausted
    """
    # Get OpenRouter model ID
    openrouter_model = get_openrouter_model_id(model)

    # Get provider
    provider = get_provider()

    # Build the specified prompt
    query = build_specified_prompt(test_case.specified_prompt, test_case.vertical)

    last_error = None
    for attempt in range(max_retries + 1):
        start_time = time.time()

        try:
            # Make the API call
            result = provider.make_grounded_call(
                prompt=query,
                model=openrouter_model,
                system_prompt=GROUNDING_SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=4096
            )

            latency_ms = (time.time() - start_time) * 1000

            # Check for empty response (some models return empty on rate limit)
            if not result.get("response_text"):
                raise Exception("Empty response received")

            response = GroundedResponse(
                response_text=result["response_text"],
                grounding_chunks=result["grounding_chunks"],
                grounding_supports=result["grounding_supports"],
                query=query,
                model=openrouter_model,
                task_id=test_case.task_id,
                criteria=[c for c in test_case.criteria],
                timestamp=time.time(),
                latency_ms=latency_ms
            )

            if output_path:
                response.save(output_path)

            return response

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Check if error is retryable
            retryable_errors = [
                "rate limit", "429", "500", "502", "503", "504",
                "timeout", "connection", "empty response", "overloaded"
            ]

            is_retryable = any(err in error_str for err in retryable_errors)

            if is_retryable and attempt < max_retries:
                delay = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"  Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
                time.sleep(delay)
            else:
                # Non-retryable error or exhausted retries
                break

    raise Exception(f"Failed after {max_retries + 1} attempts: {last_error}")


def load_test_case(path: Path) -> TestCase:
    """Load a test case from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return TestCase.from_dict(data)
