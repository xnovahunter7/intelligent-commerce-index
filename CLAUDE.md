# Visa Intelligent Commerce Index (VCI)

## Project Overview

The Visa Commerce Index (VCI) measures how well AI agents perform real-world shopping tasks. As consumers increasingly ask AI assistants to find products, compare prices, and make recommendations, VCI provides a standardized way to evaluate which agents actually get it right.

**The problem:** Most agents don't. They hallucinate prices, recommend out-of-stock items, and confidently link to products that don't exist. VCI quantifies this problem and tracks progress over time.

## Why Commerce Needs Its Own Benchmark

General AI benchmarks test reasoning and knowledge. Commerce requires something different:

- **Prices change hourly** — An answer that was right this morning might be wrong now
- **Inventory is local** — "In stock" depends on where you live
- **Details matter** — Wrong laptop specs waste $1000, not just time
- **Trust is everything** — One bad recommendation and users stop asking

VCI tests these commerce-specific skills with expert-designed tasks and grounded evaluation.

---

## The Five Verticals

Starting with five categories based on U.S. e-commerce revenue and difficulty for agents:

| Vertical | US Revenue | Why It's Hard |
|----------|------------|---------------|
| **Fashion & Apparel** | $162.9B | Size/fit is subjective, returns are expensive, style matching is fuzzy |
| **Grocery & Food** | $125.6B | Hyperlocal inventory, substitutions, dietary restrictions |
| **Electronics** | $120.1B | Specs must be exact, compatibility matters, model years are confusing |
| **Travel** | $200B+ | Dynamic pricing, multi-vendor coordination, fees hidden everywhere |
| **Home & Furniture** | $74.5B | Dimensions, delivery logistics, assembly complexity |

**Future verticals:** Beauty, Tickets & Events, Luxury, B2B

---

## Scoring System

Every task has a rubric with four types of criteria:

### 1. Hurdle (Pass/Fail)
The core requirement. If this fails, the whole task scores **zero**.

> Example: "Find a black cashmere sweater" → Agent must return something that is actually black and actually cashmere. A grey merino sweater scores 0, no matter how great the price.

### 2. Grounded (Verifiable)
Claims that can be checked against the source. This is where hallucination gets caught.
- Is that price real?
- Does that link work?
- Is it actually in stock?
- Are those specs correct?

### 3. Helpfulness (Useful)
Did the agent go beyond the minimum? Did it surface return policies, offer alternatives, flag potential issues?

### 4. Safety (Protective)
Did the agent respect dietary restrictions? Flag sketchy sellers? Avoid recommending something harmful?

### Scoring Formula

```
Score = Hurdle × (Grounded×0.4 + Helpfulness×0.3 + Safety×0.15 + Completeness×0.15)
```

Weights are configurable per vertical—electronics cares more about grounding, fashion cares more about helpfulness. These defaults are a starting point.

---

## Grounding: The Hard Part

The trickiest thing about commerce benchmarks is that the "right answer" changes constantly. A price from Tuesday is wrong by Thursday.

### Our Approach

1. **Verification window** — Grounded claims are checked within 2 hours of response generation
2. **Source matching** — Check the agent's claims against the actual pages it retrieved
3. **Staleness tracking** — Log when correct answers become incorrect, so we can refresh tasks
4. **Conservative grading** — If we can't verify a claim, it's marked unverifiable (not wrong)

---

## Category Rubrics

### Fashion & Apparel

**Hurdle:** Returns item matching category + core attributes (color, material, style)

| Criterion | Type |
|-----------|------|
| Size availability verified | Grounded |
| Material/composition accurate | Grounded |
| Price accurate (incl. shipping) | Grounded |
| Link resolves correctly | Grounded |
| Return policy surfaced | Helpfulness |
| Fit guidance provided | Helpfulness |
| Alternatives offered | Helpfulness |

### Grocery & Food

**Hurdle:** Items available for delivery to user's location

| Criterion | Type |
|-----------|------|
| Local availability verified | Grounded |
| Price/unit pricing accurate | Grounded |
| Delivery window accurate | Grounded |
| Dietary restrictions respected | Safety |
| Allergens flagged | Safety |
| Substitution logic reasonable | Helpfulness |

### Electronics

**Hurdle:** Product meets all stated technical requirements

| Criterion | Type |
|-----------|------|
| Specs match requirements | Grounded |
| Compatibility verified | Grounded |
| Model/version correct | Grounded |
| Price accurate | Grounded |
| Link resolves correctly | Grounded |
| Warranty info surfaced | Helpfulness |
| Authorized seller identified | Safety |

### Travel

**Hurdle:** Options available for requested dates and party size

| Criterion | Type |
|-----------|------|
| Dates/times accurate | Grounded |
| Total cost accurate (all fees) | Grounded |
| Availability verified | Grounded |
| Links are bookable | Grounded |
| Cancellation policy surfaced | Helpfulness |
| Multi-leg coordination works | Helpfulness |
| Entry requirements flagged | Safety |

### Home & Furniture

**Hurdle:** Item meets dimensional and style requirements

| Criterion | Type |
|-----------|------|
| Dimensions accurate | Grounded |
| Price accurate (incl. delivery) | Grounded |
| Delivery timeline accurate | Grounded |
| Link resolves correctly | Grounded |
| Assembly requirements noted | Helpfulness |
| Return policy surfaced | Helpfulness |

---

## Dataset

| Set | Tasks | Purpose |
|-----|-------|---------|
| **VCI-Dev** | 40 (8 per vertical) | Public, for development and testing |
| **VCI-Eval** | 250 (50 per vertical) | Hidden, for leaderboard scoring |

Tasks are created by domain experts (personal shoppers, category specialists, frequent buyers) and reviewed for clarity and gradability.

The eval set stays hidden to prevent overfitting. ~20% of tasks will be refreshed quarterly.

---

## Example Task

**Vertical:** Electronics
**Task ID:** VCI-ELEC-017

### Prompt
> I need a laptop for software development. Requirements: at least 16GB RAM, 512GB SSD, screen 14" or larger. Budget $1200 max. Windows preferred.

### Rubric

| # | Criterion | Type |
|---|-----------|------|
| H | Returns Windows laptop meeting all specs | Hurdle |
| 1 | RAM ≥ 16GB verified | Grounded |
| 2 | Storage ≥ 512GB SSD verified | Grounded |
| 3 | Screen ≥ 14" verified | Grounded |
| 4 | Price ≤ $1200 verified | Grounded |
| 5 | Link works | Grounded |
| 6 | Processor details accurate | Grounded |
| 7 | Developer-relevant features noted | Helpfulness |
| 8 | Warranty info provided | Helpfulness |

---

## Roadmap

### v1.0 (Current)
- 5 verticals, 290 tasks
- Leaderboard for frontier models with web search
- Open source dev set and eval harness

### v1.1 (Q2 2026)
- Add Beauty, Tickets, Luxury verticals
- Multi-turn evaluation (conversational shopping)
- Deeper failure analysis by criterion type

---

## Open Questions

Things still being figured out:

1. **Task refresh frequency?** Quarterly feels right, but popular products go in/out of stock weekly
2. **Regional expansion?** U.S. first, but EU/APAC have different merchants and regulations
3. **Live vs. cached grounding?** Real-time verification is expensive but more accurate
4. **Human baseline?** Should we measure how well humans do on these tasks for comparison?

---

## Get Involved

Looking for:
- **Domain experts** to design and review tasks
- **Model developers** to run evaluations and share feedback
- **Merchants** to validate grounding accuracy

---

## Development Notes

*Implementation based on the [Mercor ACE framework](https://github.com/Mercor-Intelligence/apex-evals/tree/main/ace) architecture.*

### Project Structure

```
vice/
├── CLAUDE.md                      # This file - project documentation
├── vci/                           # Main package
│   ├── configs/                   # Configuration and provider setup
│   │   ├── __init__.py
│   │   ├── providers.py           # Multi-provider abstraction (OpenAI, Anthropic, Gemini, Perplexity)
│   │   └── settings.py            # Global settings, vertical weights, pipeline config
│   │
│   ├── dataset/                   # CSV task definitions
│   │   ├── VCI-Fashion-dev.csv
│   │   ├── VCI-Electronics-dev.csv
│   │   ├── VCI-Grocery-dev.csv
│   │   ├── VCI-Travel-dev.csv
│   │   └── VCI-Home-dev.csv
│   │
│   ├── harness/                   # Three-stage evaluation pipeline
│   │   ├── __init__.py
│   │   ├── grounded_call.py       # Stage 1: API calls with web search
│   │   ├── grounding_pipeline.py  # Stage 2: Source scraping & mapping
│   │   └── autograder.py          # Stage 3: Claim verification & scoring
│   │
│   ├── pipeline/                  # CLI tools for running evaluations
│   │   ├── __init__.py
│   │   ├── runner.py              # Run all tasks for vertical/model/run
│   │   ├── run_all_models.py      # Launch parallel evaluation across models
│   │   ├── init_from_dataset.py   # Initialize test cases from CSV
│   │   ├── export_results.py      # Export results to CSV for analysis
│   │   └── test_single_task.py    # Test single task for setup validation
│   │
│   ├── results/                   # Output directory (generated)
│   └── tests/                     # Unit tests
│
└── docs/                          # Additional documentation
```

### Three-Stage Evaluation Pipeline

Based on the ACE architecture, VCI uses a three-stage pipeline:

#### Stage 1: Grounded API Calls (`grounded_call.py`)
- Sends prompts to AI models with web search enabled
- Captures responses with citation metadata (grounding_chunks, grounding_supports)
- Supports multiple providers: OpenAI, Anthropic, Gemini, Perplexity
- Outputs: `1_grounded_response.json`

#### Stage 2: Grounding Pipeline (`grounding_pipeline.py`)
- Extracts URLs from response (API citations + text links)
- Scrapes sources using Firecrawl API (with retries)
- Handles special content types: YouTube transcripts, Reddit posts
- Maps recommendations to supporting sources
- Outputs: `2_scraped_sources.json`

#### Stage 3: Autograder (`autograder.py`)
- Two-stage verification:
  1. **Response text validation** - Check claims explicitly stated in response
  2. **Source verification** - Cross-check against grounded sources
- Scoring system:
  - `1.0` = Criterion passes (both stages)
  - `0.0` = Fails at response text stage
  - `-1.0` = Passes response text but fails source verification
- Hurdle logic: ANY hurdle failure = task score 0
- Outputs: `3_autograder_results.json`

### Results Storage Format

```
results/{provider}/{model}/{vertical}/run_{N}/task_{ID}/
├── 0_test_case.json           # Input task definition
├── 1_grounded_response.json   # Stage 1 output
├── 2_scraped_sources.json     # Stage 2 output
└── 3_autograder_results.json  # Stage 3 output (task complete when this exists)
```

### Dataset CSV Schema

| Column | Description |
|--------|-------------|
| `Criterion ID` | Unique identifier (e.g., VCI-ELEC-001-H) |
| `Task ID` | Groups criteria into tasks (e.g., VCI-ELEC-001) |
| `Prompt` | Original user shopping request |
| `Specified Prompt` | Enhanced prompt asking for explicit details |
| `Vertical` | Category (fashion, electronics, grocery, travel, home) |
| `Workflow` | Task type (Bargain Hunting, Compatibility, etc.) |
| `Hurdle Tag` | "Hurdle" for must-pass, "Not" for supporting criteria |
| `Criteria type` | What's being evaluated (Pricing, Product specs, etc.) |
| `Criterion Grounding Check` | "Grounded" or "Not Grounded" |
| `Description` | Specific success criterion |
| `Shop vs. Product` | Classification for evaluation routing |

### Per-Vertical Scoring Weights

Configured in `vci/configs/settings.py`:

| Vertical | Grounded | Helpfulness | Safety | Completeness |
|----------|----------|-------------|--------|--------------|
| Fashion | 0.35 | 0.35 | 0.15 | 0.15 |
| Grocery | 0.35 | 0.25 | **0.25** | 0.15 |
| Electronics | **0.45** | 0.25 | 0.15 | 0.15 |
| Travel | 0.40 | 0.30 | 0.15 | 0.15 |
| Home | 0.40 | 0.30 | 0.10 | **0.20** |

### CLI Usage

```bash
# Initialize test cases from CSV
python -m vci.pipeline.init_from_dataset --vertical electronics --dataset dev

# Test single task
python -m vci.pipeline.test_single_task --task VCI-ELEC-001 --model gpt-4o

# Run all tasks for a vertical/model
python -m vci.pipeline.runner --vertical electronics --model gpt-4o --run 1

# Run all models in parallel
python -m vci.pipeline.run_all_models --vertical electronics --run 1

# Export results to CSV
python -m vci.pipeline.export_results --output results_summary.csv --aggregate
```

### Environment Variables

```bash
# Required - Get your key at https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-v1-...

# Optional
VCI_SITE_NAME="VCI Benchmark"  # Shows in OpenRouter dashboard
```

### Supported Models (via OpenRouter)

All models are accessed through OpenRouter. Use either short names or full OpenRouter IDs:

| Short Name | OpenRouter ID | Notes |
|------------|---------------|-------|
| `gpt-4o` | `openai/gpt-4o` | |
| `gpt-4o-mini` | `openai/gpt-4o-mini` | |
| `claude-3.5-sonnet` | `anthropic/claude-3.5-sonnet` | |
| `claude-3-opus` | `anthropic/claude-3-opus` | |
| `gemini-2.0-flash` | `google/gemini-2.0-flash-001` | |
| `gemini-1.5-pro` | `google/gemini-pro-1.5` | |
| `sonar-pro` | `perplexity/sonar-pro` | Built-in web search |
| `sonar` | `perplexity/sonar` | Built-in web search |
| `llama-3.1-70b` | `meta-llama/llama-3.1-70b-instruct` | |
| `deepseek-chat` | `deepseek/deepseek-chat` | |

You can also use any OpenRouter model ID directly (e.g., `openai/gpt-4o-2024-08-06`).

---

## Quick Start

### 1. Setup

```bash
cd vice

# Install dependencies
pip install -r requirements.txt

# Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

### 2. Run a single vertical

```bash
# Run electronics tasks with GPT-4o
python -m vci.pipeline.runner --vertical electronics --model gpt-4o --run 1

# Run fashion tasks with Claude
python -m vci.pipeline.runner --vertical fashion --model claude-3.5-sonnet --run 1

# Run with Perplexity (has built-in web search)
python -m vci.pipeline.runner --vertical grocery --model sonar-pro --run 1
```

### 3. View results

Results are saved to `results/{model}/{vertical}/run_{N}/task_{ID}/`:

```bash
# View a response
cat results/openai_gpt-4o/electronics/run_1/task_VCI-ELEC-001/1_grounded_response.json | jq .response_text
```

### Available Verticals

- `fashion` - Apparel, accessories, size/fit
- `electronics` - Laptops, phones, specs
- `grocery` - Food delivery, dietary restrictions
- `travel` - Flights, hotels, booking
- `home` - Furniture, dimensions, delivery

---

### Key Implementation Considerations

- **OpenRouter unified access** — Single API key for all models
- **Grounding verification** must happen within 2-hour window
- **Conservative scoring** — unverifiable != wrong (configurable)
- **Per-vertical weight configuration** for scoring formula
- **Parallel processing** — 100 workers for scraping/grading

### TODO: Implementation Gaps

The following need to be implemented:

1. **Scraping implementations** in `harness/grounding_pipeline.py`:
   - `firecrawl_scrape()` - Main webpage scraping
   - `scrape_youtube_transcript()` - YouTube video handling
   - `scrape_reddit_post()` - Reddit content extraction

2. **LLM grading** in `harness/autograder.py`:
   - `grade_response_text()` - Stage 1 validation prompts
   - `grade_against_sources()` - Stage 2 verification prompts

3. **Dataset expansion**:
   - Add more tasks to reach 8 per vertical (dev) / 50 per vertical (eval)
   - Create VCI-*-eval.csv files (hidden evaluation sets)
