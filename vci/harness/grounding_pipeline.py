"""
VCI Grounding Pipeline

Stage 2 of Pipeline: Scrapes cited sources and maps recommendations to sources.
Handles various content types and builds verification context.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from ..configs.settings import SETTINGS


@dataclass
class ScrapedSource:
    """A scraped source with content."""
    url: str
    title: str
    content: str  # Markdown content
    content_type: str  # webpage, youtube, reddit, etc.
    scrape_timestamp: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProductSourceMap:
    """Maps a product recommendation to its supporting sources."""
    product_name: str
    product_index: int
    source_indices: List[int]  # Indices into scraped_sources
    extracted_claims: Dict[str, Any]  # Claims about this product from response


@dataclass
class GroundingResult:
    """Result of the grounding pipeline."""
    task_id: str
    recommendations: List[str]  # Extracted product/recommendation names
    scraped_sources: List[ScrapedSource]
    product_source_map: List[ProductSourceMap]
    failed_scrapes: List[Dict[str, str]]  # URLs that failed
    timestamp: float
    total_latency_ms: float

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "recommendations": self.recommendations,
            "scraped_sources": [s.to_dict() for s in self.scraped_sources],
            "product_source_map": [asdict(m) for m in self.product_source_map],
            "failed_scrapes": self.failed_scrapes,
            "timestamp": self.timestamp,
            "total_latency_ms": self.total_latency_ms
        }

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def extract_urls_from_response(grounded_response: dict) -> List[str]:
    """
    Extract all URLs from a grounded response.

    Sources:
    1. grounding_chunks from API (citations)
    2. URLs mentioned in response_text
    """
    urls = set()

    # From grounding chunks
    for chunk in grounded_response.get("grounding_chunks", []):
        if "url" in chunk:
            urls.add(chunk["url"])
        if "uri" in chunk:
            urls.add(chunk["uri"])

    # TODO: Extract URLs from response_text using regex
    # import re
    # url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    # urls.update(re.findall(url_pattern, grounded_response.get("response_text", "")))

    return list(urls)


def classify_url(url: str) -> str:
    """Classify URL by content type for specialized scraping."""
    domain = urlparse(url).netloc.lower()

    if "youtube.com" in domain or "youtu.be" in domain:
        return "youtube"
    elif "reddit.com" in domain:
        return "reddit"
    elif "amazon.com" in domain:
        return "amazon"
    elif "walmart.com" in domain:
        return "walmart"
    elif "target.com" in domain:
        return "target"
    elif "bestbuy.com" in domain:
        return "bestbuy"
    else:
        return "webpage"


def scrape_url(url: str, retries: int = None, timeout: int = 30) -> ScrapedSource:
    """
    Scrape a URL and return structured content.

    Uses requests + BeautifulSoup for general web scraping.
    Handles various content types with specialized parsing.
    """
    if retries is None:
        retries = SETTINGS.max_scrape_retries

    content_type = classify_url(url)
    start_time = time.time()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract title
                title = ""
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text(strip=True)

                # Remove script/style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()

                # Extract product-specific info for e-commerce sites
                content_parts = []

                # Try to find product info
                product_info = extract_product_info(soup, content_type)
                if product_info:
                    content_parts.append(product_info)

                # Get main content
                main_content = soup.find("main") or soup.find("article") or soup.find("body")
                if main_content:
                    text = main_content.get_text(separator="\n", strip=True)
                    # Limit content size
                    text = text[:10000] if len(text) > 10000 else text
                    content_parts.append(text)

                content = "\n\n".join(content_parts)

                return ScrapedSource(
                    url=url,
                    title=title,
                    content=content,
                    content_type=content_type,
                    scrape_timestamp=time.time(),
                    success=True,
                    error=None
                )

            elif response.status_code in [403, 429]:
                last_error = f"HTTP {response.status_code}: Access denied or rate limited"
                time.sleep(1 * (attempt + 1))  # Backoff
            else:
                last_error = f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            last_error = "Request timeout"
        except requests.exceptions.ConnectionError:
            last_error = "Connection error"
        except Exception as e:
            last_error = str(e)

        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))

    return ScrapedSource(
        url=url,
        title="",
        content="",
        content_type=content_type,
        scrape_timestamp=time.time(),
        success=False,
        error=last_error
    )


def extract_product_info(soup: BeautifulSoup, content_type: str) -> Optional[str]:
    """Extract structured product information from e-commerce pages."""
    info_parts = []

    # Price extraction patterns
    price_selectors = [
        {"class": re.compile(r"price", re.I)},
        {"itemprop": "price"},
        {"data-testid": re.compile(r"price", re.I)},
    ]

    for selector in price_selectors:
        price_elem = soup.find(attrs=selector)
        if price_elem:
            price_text = price_elem.get_text(strip=True)
            if "$" in price_text or "price" in price_text.lower():
                info_parts.append(f"Price: {price_text}")
                break

    # Product name
    name_selectors = [
        {"itemprop": "name"},
        {"class": re.compile(r"product.*title|product.*name", re.I)},
        {"data-testid": re.compile(r"product.*title", re.I)},
    ]

    for selector in name_selectors:
        name_elem = soup.find(attrs=selector)
        if name_elem:
            info_parts.append(f"Product: {name_elem.get_text(strip=True)}")
            break

    # Availability
    avail_selectors = [
        {"class": re.compile(r"availability|stock|in.stock", re.I)},
        {"itemprop": "availability"},
    ]

    for selector in avail_selectors:
        avail_elem = soup.find(attrs=selector)
        if avail_elem:
            info_parts.append(f"Availability: {avail_elem.get_text(strip=True)}")
            break

    # Material/composition for fashion
    if content_type in ["webpage", "amazon", "walmart", "target"]:
        material_patterns = [
            re.compile(r"(material|composition|fabric)[:\s]*([^<\n]+)", re.I),
            re.compile(r"(\d+%\s*\w+(?:\s*,\s*\d+%\s*\w+)*)", re.I),
        ]
        page_text = soup.get_text()
        for pattern in material_patterns:
            match = pattern.search(page_text)
            if match:
                info_parts.append(f"Material: {match.group(0)[:200]}")
                break

    return "\n".join(info_parts) if info_parts else None


def extract_recommendations(response_text: str, vertical: str) -> List[str]:
    """
    Extract product/recommendation names from response text.

    Uses pattern matching to identify product names from common formats:
    - **Product Name** (bold markdown)
    - 1. Product Name
    - - Product Name
    - "Product Name"
    """
    recommendations = []

    # Pattern 1: Bold markdown **Product Name**
    bold_pattern = re.compile(r'\*\*([^*]+)\*\*')
    for match in bold_pattern.findall(response_text):
        # Filter out non-product bold text
        if len(match) > 5 and len(match) < 100 and not match.lower().startswith(("exact", "available", "material", "return", "note")):
            recommendations.append(match.strip())

    # Pattern 2: Numbered lists with product names "1. Product Name"
    numbered_pattern = re.compile(r'^\d+\.\s*\*?\*?([^*\n]+?)(?:\*?\*?[\s-]*(?:$|\[|http|\(|:))', re.MULTILINE)
    for match in numbered_pattern.findall(response_text):
        name = match.strip().rstrip(':-')
        if name and len(name) > 3 and len(name) < 100:
            recommendations.append(name)

    # Pattern 3: Product/brand name followed by price pattern
    price_pattern = re.compile(r'([A-Z][^.!?\n]{5,50}?)(?:\s*[-–]\s*|\s+)?\$\d+', re.MULTILINE)
    for match in price_pattern.findall(response_text):
        name = match.strip()
        if name and not any(word in name.lower() for word in ["under", "over", "around", "about"]):
            recommendations.append(name)

    # Deduplicate while preserving order
    seen = set()
    unique_recs = []
    for rec in recommendations:
        rec_lower = rec.lower()
        if rec_lower not in seen:
            seen.add(rec_lower)
            unique_recs.append(rec)

    return unique_recs[:10]  # Limit to 10 recommendations


def map_sources_to_products(
    recommendations: List[str],
    sources: List[ScrapedSource],
    response_text: str
) -> List[ProductSourceMap]:
    """
    Map which sources support which product recommendations.

    Uses text matching to find which URLs are associated with which products
    by analyzing proximity in the response text.
    """
    product_maps = []

    for i, rec in enumerate(recommendations):
        source_indices = []
        extracted_claims = {}

        # Find where this product is mentioned in response
        rec_lower = rec.lower()
        rec_pos = response_text.lower().find(rec_lower)

        if rec_pos >= 0:
            # Look for URLs near this product mention (within 500 chars)
            context_start = max(0, rec_pos - 100)
            context_end = min(len(response_text), rec_pos + 500)
            context = response_text[context_start:context_end]

            # Find URLs in context
            url_pattern = re.compile(r'https?://[^\s<>"\'\)]+')
            urls_in_context = url_pattern.findall(context)

            # Map to source indices
            for url in urls_in_context:
                for j, source in enumerate(sources):
                    if url.rstrip('/') == source.url.rstrip('/'):
                        if j not in source_indices:
                            source_indices.append(j)

            # Extract claims from context
            # Price
            price_match = re.search(r'\$[\d,]+(?:\.\d{2})?', context)
            if price_match:
                extracted_claims["price"] = price_match.group(0)

            # Material
            material_match = re.search(r'(?:100%|[\d]+%)\s*(?:cashmere|cotton|wool|polyester|silk|linen)', context, re.I)
            if material_match:
                extracted_claims["material"] = material_match.group(0)

        product_maps.append(ProductSourceMap(
            product_name=rec,
            product_index=i,
            source_indices=source_indices,
            extracted_claims=extracted_claims
        ))

    return product_maps


def run_grounding_pipeline(
    grounded_response: dict,
    vertical: str,
    output_path: Optional[Path] = None
) -> GroundingResult:
    """
    Run the full grounding pipeline.

    1. Extract URLs from response
    2. Scrape all sources in parallel
    3. Extract product recommendations
    4. Map sources to products

    Args:
        grounded_response: Output from Stage 1 (grounded_call)
        vertical: The vertical (fashion, electronics, etc.)
        output_path: Optional path to save results

    Returns:
        GroundingResult with scraped sources and mappings
    """
    start_time = time.time()
    task_id = grounded_response.get("task_id", "unknown")

    # Extract URLs
    urls = extract_urls_from_response(grounded_response)

    # Scrape in parallel
    scraped_sources = []
    failed_scrapes = []

    with ThreadPoolExecutor(max_workers=SETTINGS.max_workers) as executor:
        future_to_url = {executor.submit(scrape_url, url): url for url in urls}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                source = future.result()
                if source.success:
                    scraped_sources.append(source)
                else:
                    failed_scrapes.append({"url": url, "error": source.error})
            except Exception as e:
                failed_scrapes.append({"url": url, "error": str(e)})

    # Extract recommendations
    response_text = grounded_response.get("response_text", "")
    recommendations = extract_recommendations(response_text, vertical)

    # Map sources to products
    product_source_map = map_sources_to_products(
        recommendations, scraped_sources, response_text
    )

    total_latency_ms = (time.time() - start_time) * 1000

    result = GroundingResult(
        task_id=task_id,
        recommendations=recommendations,
        scraped_sources=scraped_sources,
        product_source_map=product_source_map,
        failed_scrapes=failed_scrapes,
        timestamp=time.time(),
        total_latency_ms=total_latency_ms
    )

    if output_path:
        result.save(output_path)

    return result
