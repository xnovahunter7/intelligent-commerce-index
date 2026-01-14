"""
VCI Autograder Module

Stage 3 of Pipeline: Verifies claims and scores responses.
Implements two-stage verification: response text validation, then source grounding.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests

from ..configs.settings import SETTINGS, VERTICAL_WEIGHTS, ScoringWeights


# Grader model - use a fast, capable model for judging
GRADER_MODEL = os.environ.get("VCI_GRADER_MODEL", "anthropic/claude-3.5-sonnet")


class CriterionType(Enum):
    """Types of criteria in the VCI rubric."""
    HURDLE = "hurdle"
    GROUNDED = "grounded"
    HELPFULNESS = "helpfulness"
    SAFETY = "safety"
    COMPLETENESS = "completeness"


class GradingStage(Enum):
    """Stages at which grading occurs."""
    RESPONSE_TEXT = "response_text"
    LINK_VERIFICATION = "link_verification"
    GROUNDED_SOURCES = "grounded_sources"


class EvaluationType(Enum):
    """How to evaluate criteria across products."""
    PER_PRODUCT_ALL = "per_product_all"  # ALL products must pass
    PER_PRODUCT_ANY = "per_product_any"  # At least N products must pass
    HOLISTIC = "holistic"  # Evaluate response as a whole


@dataclass
class CriterionResult:
    """Result of grading a single criterion."""
    criterion_id: str
    criterion_type: CriterionType
    description: str
    passed: bool
    score: float  # 1.0 = pass, 0.0 = fail response text, -1.0 = fail source verification
    stage_reached: GradingStage
    reasoning: str
    product_results: List[Dict[str, Any]] = field(default_factory=list)
    verification_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["criterion_type"] = self.criterion_type.value
        result["stage_reached"] = self.stage_reached.value
        return result


@dataclass
class TaskResult:
    """Complete grading result for a task."""
    task_id: str
    vertical: str
    hurdle_passed: bool
    total_score: float
    component_scores: Dict[str, float]  # grounded, helpfulness, safety, completeness
    criterion_results: List[CriterionResult]
    weights_used: ScoringWeights
    timestamp: float
    total_latency_ms: float

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "vertical": self.vertical,
            "hurdle_passed": self.hurdle_passed,
            "total_score": self.total_score,
            "component_scores": self.component_scores,
            "criterion_results": [c.to_dict() for c in self.criterion_results],
            "weights_used": asdict(self.weights_used),
            "timestamp": self.timestamp,
            "total_latency_ms": self.total_latency_ms
        }

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def determine_evaluation_type(description: str) -> Tuple[EvaluationType, int]:
    """
    Determine evaluation type from criterion description.

    Returns (evaluation_type, required_pass_count)
    - required_pass_count = -1 means ALL must pass
    """
    description_lower = description.lower()

    # Check for quantity patterns
    if any(word in description_lower for word in ["all", "every", "each", "only"]):
        return EvaluationType.PER_PRODUCT_ALL, -1

    if "at least" in description_lower:
        # Extract number: "at least 2 products"
        import re
        match = re.search(r"at least (\d+)", description_lower)
        if match:
            return EvaluationType.PER_PRODUCT_ANY, int(match.group(1))
        return EvaluationType.PER_PRODUCT_ANY, 1

    if any(word in description_lower for word in ["provides", "returns", "gives"]):
        # Quantity check - holistic
        return EvaluationType.HOLISTIC, 1

    # Default: all products must pass
    return EvaluationType.PER_PRODUCT_ALL, -1


def call_grader_llm(prompt: str, max_tokens: int = 1024) -> str:
    """Call the grader LLM via OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://vci-benchmark.com",
        "X-Title": "VCI Autograder",
    }

    payload = {
        "model": GRADER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,  # Deterministic grading
        "max_tokens": max_tokens,
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=180  # 3 minutes for complex grading
    )

    if response.status_code != 200:
        raise Exception(f"Grader API error: {response.status_code} - {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def grade_response_text(
    criterion: Dict[str, Any],
    response_text: str,
    recommendations: List[str]
) -> Tuple[bool, str, List[Dict]]:
    """
    Stage 1: Validate claims in response text.

    Rules:
    - Zero background knowledge: judge only what's explicitly stated
    - Per-product scope: evaluate text describing each specific product
    - No vacuous truth: if no products recommended, feature criteria fail
    """
    description = criterion.get("description", "")
    criterion_type = criterion.get("type", "grounded")

    # No products = fail feature criteria
    if not recommendations and criterion_type != "hurdle":
        return False, "No products recommended in response", []

    # Build grading prompt
    prompt = f"""You are a strict grader evaluating an AI shopping assistant's response.

CRITERION TO EVALUATE:
{description}

RESPONSE TO EVALUATE:
{response_text}

GRADING RULES:
1. ONLY judge what is EXPLICITLY stated in the response text
2. Do NOT use your own knowledge - if information isn't in the response, it's NOT there
3. Implied or inferred information does NOT count
4. Be strict but fair

RESPONSE FORMAT:
Return ONLY a JSON object with these fields:
{{
    "passed": true/false,
    "reasoning": "Brief explanation of your judgment",
    "evidence": "Quote from response that supports your judgment (or 'N/A' if criterion fails)"
}}

Evaluate now:"""

    try:
        result_text = call_grader_llm(prompt)

        # Parse JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{[^{}]*"passed"[^{}]*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result.get("passed", False), result.get("reasoning", ""), []
        else:
            # Try to determine pass/fail from text
            passed = "pass" in result_text.lower() and "fail" not in result_text.lower()
            return passed, result_text[:500], []

    except Exception as e:
        return False, f"Grading error: {str(e)}", []


def verify_link(url: str, expected_type: str = "product", timeout: int = 10) -> Tuple[bool, str]:
    """
    Verify that a link resolves to valid content.

    For shopping: must be a valid product/purchase page.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)

        if response.status_code == 200:
            return True, "Link resolves successfully"
        elif response.status_code == 405:
            # HEAD not allowed, try GET
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
            if response.status_code == 200:
                return True, "Link resolves successfully (via GET)"
            return False, f"HTTP {response.status_code}"
        elif response.status_code in [301, 302, 307, 308]:
            return True, f"Link redirects (HTTP {response.status_code})"
        elif response.status_code == 403:
            # Many e-commerce sites block bots but page likely exists
            return True, "Link likely valid (403 bot protection)"
        elif response.status_code == 404:
            return False, "Page not found (404)"
        else:
            return False, f"HTTP {response.status_code}"

    except requests.exceptions.Timeout:
        return False, "Request timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection error"
    except Exception as e:
        return False, str(e)


def grade_against_sources(
    criterion: Dict[str, Any],
    product_source_map: List[Dict],
    scraped_sources: List[Dict],
    response_text: str
) -> Tuple[bool, str, Dict]:
    """
    Stage 2: Verify claims against grounded sources.

    For products with mapped sources:
    - Retrieve content from scraped sources
    - Confirm criterion truth in at least one source
    """
    description = criterion.get("description", "")
    verification_details = {
        "sources_checked": 0,
        "sources_supporting": 0,
        "source_details": []
    }

    # Collect all source content
    source_contents = []
    for mapping in product_source_map:
        for idx in mapping.get("source_indices", []):
            if idx < len(scraped_sources):
                source = scraped_sources[idx]
                if source.get("success") and source.get("content"):
                    source_contents.append({
                        "url": source.get("url", ""),
                        "title": source.get("title", ""),
                        "content": source.get("content", "")[:3000]  # Limit content
                    })

    if not source_contents:
        return False, "No source content available for verification", verification_details

    # Build source verification prompt
    sources_text = "\n\n---\n\n".join([
        f"SOURCE: {s['url']}\nTITLE: {s['title']}\nCONTENT:\n{s['content']}"
        for s in source_contents[:3]  # Limit to 3 sources
    ])

    prompt = f"""You are verifying claims made by an AI shopping assistant against source documents.

CRITERION TO VERIFY:
{description}

CLAIM FROM AI RESPONSE:
{response_text[:2000]}

SOURCE DOCUMENTS:
{sources_text}

TASK:
Determine if the criterion is SUPPORTED by the source documents.
The claim must be explicitly verifiable in at least one source.

RESPONSE FORMAT:
Return ONLY a JSON object:
{{
    "supported": true/false,
    "reasoning": "Brief explanation",
    "supporting_source": "URL of supporting source or 'none'"
}}

Verify now:"""

    try:
        result_text = call_grader_llm(prompt, max_tokens=512)

        json_match = re.search(r'\{[^{}]*"supported"[^{}]*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            verification_details["sources_checked"] = len(source_contents)
            verification_details["sources_supporting"] = 1 if result.get("supported") else 0
            return result.get("supported", False), result.get("reasoning", ""), verification_details
        else:
            supported = "support" in result_text.lower() and "not support" not in result_text.lower()
            return supported, result_text[:500], verification_details

    except Exception as e:
        return False, f"Source verification error: {str(e)}", verification_details


def map_criteria_type(criterion: Dict[str, Any]) -> CriterionType:
    """
    Map criterion from CSV to CriterionType enum.

    CSV has:
    - "type": "hurdle" or "grounded" (from Hurdle Tag)
    - "criteria_type": "Helpfulness", "Safety", "Product specs", "Comparison", "Risk identification", etc.

    We need to properly map these to our enum values.
    """
    # If it's a hurdle, that takes precedence
    if criterion.get("type", "").lower() == "hurdle":
        return CriterionType.HURDLE

    # Map criteria_type from CSV to our enum
    csv_type = criterion.get("criteria_type", "").lower()

    if "helpfulness" in csv_type:
        return CriterionType.HELPFULNESS
    elif "comparison" in csv_type:
        # Comparison quality is a form of helpfulness (going beyond minimum)
        return CriterionType.HELPFULNESS
    elif "risk" in csv_type:
        # Risk identification is safety-adjacent but also helpfulness
        return CriterionType.HELPFULNESS
    elif "safety" in csv_type:
        return CriterionType.SAFETY
    elif "completeness" in csv_type:
        return CriterionType.COMPLETENESS
    else:
        # Default: Product specs, Pricing, Link validity, Core requirement -> GROUNDED
        return CriterionType.GROUNDED


def grade_link_criterion(
    criterion: Dict[str, Any],
    response_text: str,
    scraped_sources: List[Dict],
    failed_scrapes: List[Dict]
) -> CriterionResult:
    """
    Grade a link validity criterion by checking if URLs resolve.

    Uses both successful scrapes and explicit link verification.
    """
    criterion_id = criterion.get("criterion_id", "unknown")
    criterion_type = map_criteria_type(criterion)
    description = criterion.get("description", "")

    # Extract URLs from response text
    url_pattern = re.compile(r'https?://[^\s<>"\'\)]+')
    urls_in_response = url_pattern.findall(response_text)

    if not urls_in_response:
        return CriterionResult(
            criterion_id=criterion_id,
            criterion_type=criterion_type,
            description=description,
            passed=False,
            score=0.0,
            stage_reached=GradingStage.LINK_VERIFICATION,
            reasoning="No URLs found in response"
        )

    # Check against scraped sources and failed scrapes
    successful_urls = {s.get("url", "").rstrip("/") for s in scraped_sources if s.get("success")}
    failed_urls = {f.get("url", "").rstrip("/") for f in failed_scrapes}

    verified_count = 0
    failed_count = 0
    unverified_urls = []

    for url in urls_in_response:
        url_normalized = url.rstrip("/")
        if url_normalized in successful_urls:
            verified_count += 1
        elif url_normalized in failed_urls:
            failed_count += 1
        else:
            # URL wasn't scraped, try direct verification
            link_ok, reason = verify_link(url)
            if link_ok:
                verified_count += 1
            else:
                failed_count += 1
                unverified_urls.append(f"{url}: {reason}")

    # Score based on verification results
    total_urls = len(urls_in_response)

    if failed_count > 0:
        # At least one link failed
        score = max(0, (verified_count - failed_count) / total_urls)
        passed = False
        reasoning = f"{failed_count}/{total_urls} links failed verification"
        if unverified_urls:
            reasoning += f". Failed: {unverified_urls[:3]}"
    else:
        score = 1.0
        passed = True
        reasoning = f"All {verified_count} links verified successfully"

    return CriterionResult(
        criterion_id=criterion_id,
        criterion_type=criterion_type,
        description=description,
        passed=passed,
        score=score,
        stage_reached=GradingStage.LINK_VERIFICATION,
        reasoning=reasoning,
        verification_details={
            "total_urls": total_urls,
            "verified": verified_count,
            "failed": failed_count
        }
    )


def grade_criterion(
    criterion: Dict[str, Any],
    response_text: str,
    recommendations: List[str],
    product_source_map: List[Dict],
    scraped_sources: List[Dict],
    failed_scrapes: List[Dict] = None
) -> CriterionResult:
    """
    Grade a single criterion through the two-stage process.
    """
    if failed_scrapes is None:
        failed_scrapes = []

    criterion_id = criterion.get("criterion_id", "unknown")
    criterion_type = map_criteria_type(criterion)
    description = criterion.get("description", "")
    is_grounded = criterion.get("grounded_status", "Grounded") == "Grounded"

    # Special handling for link criteria - use dedicated link grader
    if "link" in description.lower() and any(word in description.lower() for word in ["works", "valid", "resolves", "correct"]):
        return grade_link_criterion(criterion, response_text, scraped_sources, failed_scrapes)

    # Stage 1: Response text validation
    stage1_passed, stage1_reasoning, product_results = grade_response_text(
        criterion, response_text, recommendations
    )

    if not stage1_passed:
        return CriterionResult(
            criterion_id=criterion_id,
            criterion_type=criterion_type,
            description=description,
            passed=False,
            score=0.0,
            stage_reached=GradingStage.RESPONSE_TEXT,
            reasoning=stage1_reasoning,
            product_results=product_results
        )

    # Non-grounded criteria (helpfulness, safety, etc.): done after stage 1
    if not is_grounded:
        return CriterionResult(
            criterion_id=criterion_id,
            criterion_type=criterion_type,
            description=description,
            passed=True,
            score=1.0,
            stage_reached=GradingStage.RESPONSE_TEXT,
            reasoning=stage1_reasoning,
            product_results=product_results
        )

    # Stage 2: Source verification for grounded criteria
    stage2_passed, stage2_reasoning, verification_details = grade_against_sources(
        criterion, product_source_map, scraped_sources, response_text
    )

    # Track if failure was due to scraping issues vs actual hallucination
    # Check both success flag AND content quality (blocked pages have minimal content)
    def is_useful_content(source: Dict) -> bool:
        if not source.get("success"):
            return False
        content = source.get("content", "")
        # Blocked pages typically have very short content or contain bot detection phrases
        blocked_indicators = [
            "continue shopping",
            "access denied",
            "robot check",
            "captcha",
            "please verify",
            "blocked",
        ]
        if len(content) < 500:  # Too short to be useful product page
            return False
        content_lower = content.lower()
        if any(indicator in content_lower for indicator in blocked_indicators):
            return False
        return True

    useful_sources = [s for s in scraped_sources if is_useful_content(s)]
    scrape_failure = len(useful_sources) == 0

    if not stage2_passed:
        # Determine score based on failure type
        if scrape_failure:
            # Infrastructure failure - don't penalize model
            return CriterionResult(
                criterion_id=criterion_id,
                criterion_type=criterion_type,
                description=description,
                passed=False,
                score=0.0,  # Neutral - couldn't verify
                stage_reached=GradingStage.GROUNDED_SOURCES,
                reasoning=f"Scrape failure - unverifiable: {stage2_reasoning}",
                verification_details={**verification_details, "scrape_failure": True}
            )
        else:
            # Sources available but claim not supported - potential hallucination
            return CriterionResult(
                criterion_id=criterion_id,
                criterion_type=criterion_type,
                description=description,
                passed=False,
                score=-1.0,  # Penalty for hallucination
                stage_reached=GradingStage.GROUNDED_SOURCES,
                reasoning=f"Not grounded in sources: {stage2_reasoning}",
                verification_details={**verification_details, "hallucination": True}
            )

    return CriterionResult(
        criterion_id=criterion_id,
        criterion_type=criterion_type,
        description=description,
        passed=True,
        score=1.0,
        stage_reached=GradingStage.GROUNDED_SOURCES,
        reasoning=stage2_reasoning,
        product_results=product_results,
        verification_details=verification_details
    )


def calculate_final_score(
    criterion_results: List[CriterionResult],
    weights: ScoringWeights
) -> Tuple[bool, float, Dict[str, float]]:
    """
    Calculate final task score using the VCI formula.

    Score = Hurdle × (Grounded×w1 + Helpfulness×w2 + Safety×w3 + Completeness×w4)

    Returns: (hurdle_passed, total_score, component_scores)
    """
    # Check hurdle
    hurdle_results = [c for c in criterion_results if c.criterion_type == CriterionType.HURDLE]
    hurdle_passed = all(c.passed for c in hurdle_results) if hurdle_results else True

    if not hurdle_passed:
        return False, 0.0, {"grounded": 0, "helpfulness": 0, "safety": 0, "completeness": 0}

    # Calculate component scores
    def avg_score(ctype: CriterionType) -> float:
        results = [c for c in criterion_results if c.criterion_type == ctype]
        if not results:
            return 1.0  # No criteria of this type = full score
        return sum(max(0, c.score) for c in results) / len(results)

    component_scores = {
        "grounded": avg_score(CriterionType.GROUNDED),
        "helpfulness": avg_score(CriterionType.HELPFULNESS),
        "safety": avg_score(CriterionType.SAFETY),
        "completeness": avg_score(CriterionType.COMPLETENESS)
    }

    # Apply weights
    total_score = (
        component_scores["grounded"] * weights.grounded +
        component_scores["helpfulness"] * weights.helpfulness +
        component_scores["safety"] * weights.safety +
        component_scores["completeness"] * weights.completeness
    )

    return hurdle_passed, total_score, component_scores


def grade_response(
    grounded_response: dict,
    grounding_result: dict,
    criteria: List[Dict[str, Any]],
    vertical: str,
    output_path: Optional[Path] = None
) -> TaskResult:
    """
    Grade a complete response.

    Args:
        grounded_response: Output from Stage 1
        grounding_result: Output from Stage 2
        criteria: List of criteria to grade against
        vertical: The vertical (for weight selection)
        output_path: Optional path to save results

    Returns:
        TaskResult with complete grading
    """
    start_time = time.time()
    task_id = grounded_response.get("task_id", "unknown")

    response_text = grounded_response.get("response_text", "")
    recommendations = grounding_result.get("recommendations", [])
    product_source_map = grounding_result.get("product_source_map", [])
    scraped_sources = grounding_result.get("scraped_sources", [])
    failed_scrapes = grounding_result.get("failed_scrapes", [])

    # Grade criteria in parallel
    criterion_results = []

    with ThreadPoolExecutor(max_workers=SETTINGS.max_workers) as executor:
        futures = {
            executor.submit(
                grade_criterion,
                criterion,
                response_text,
                recommendations,
                product_source_map,
                scraped_sources,
                failed_scrapes
            ): criterion
            for criterion in criteria
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                criterion_results.append(result)
            except Exception as e:
                criterion = futures[future]
                criterion_results.append(CriterionResult(
                    criterion_id=criterion.get("criterion_id", "unknown"),
                    criterion_type=CriterionType.GROUNDED,
                    description=criterion.get("description", ""),
                    passed=False,
                    score=0.0,
                    stage_reached=GradingStage.RESPONSE_TEXT,
                    reasoning=f"Grading error: {str(e)}"
                ))

    # Sort by criterion_id to maintain order
    criterion_results.sort(key=lambda x: x.criterion_id)

    # Get weights for vertical
    weights = VERTICAL_WEIGHTS.get(vertical, ScoringWeights())

    # Calculate final score
    hurdle_passed, total_score, component_scores = calculate_final_score(
        criterion_results, weights
    )

    total_latency_ms = (time.time() - start_time) * 1000

    result = TaskResult(
        task_id=task_id,
        vertical=vertical,
        hurdle_passed=hurdle_passed,
        total_score=total_score,
        component_scores=component_scores,
        criterion_results=criterion_results,
        weights_used=weights,
        timestamp=time.time(),
        total_latency_ms=total_latency_ms
    )

    if output_path:
        result.save(output_path)

    return result
