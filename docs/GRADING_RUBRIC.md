# VCI Grading Rubric

## Overview

The Visa Intelligent Commerce Index (VCI) evaluates AI shopping assistants on their ability to provide **accurate, verifiable, and helpful** product recommendations. The scoring system is designed to measure real-world usefulness for consumers making purchasing decisions.

## Three-Stage Evaluation Pipeline

### Stage 1: Grounded Response
The AI model receives a shopping query and must provide recommendations with:
- Product names and descriptions
- Exact prices
- Availability information
- Direct purchase links
- Relevant specifications

### Stage 2: Source Verification
All URLs and claims are verified:
- Links are checked to confirm they resolve
- Product pages are scraped for actual content
- Claims in the response are mapped to source evidence

### Stage 3: Automated Grading
An LLM judge evaluates each criterion using strict rules:
- **Zero background knowledge**: Only explicitly stated information counts
- **No inference**: Implied or assumed information fails
- **Per-product evaluation**: Each recommendation is assessed independently

---

## Scoring Formula

```
VCI Score = Hurdle × (Grounded × w₁ + Helpfulness × w₂ + Safety × w₃ + Completeness × w₄)
```

| Component | Weight | Description |
|-----------|--------|-------------|
| Hurdle | Pass/Fail | Must pass to score any points |
| Grounded | 35-45% | Verified factual accuracy |
| Helpfulness | 25-35% | Useful additional information |
| Safety | 10-25% | No harmful recommendations |
| Completeness | 15-20% | All requested info provided |

*Weights vary by vertical (e.g., Electronics emphasizes Grounded; Grocery emphasizes Safety)*

---

## Criterion Types

### Hurdle Criteria (Pass/Fail Gate)
**If any hurdle fails, the entire task scores zero.**

Examples:
- "Returns item that is black and cashmere" ✓/✗
- "Returns formal dress in navy or burgundy in size 8" ✓/✗
- "Returns waterproof hiking boots with ankle support" ✓/✗

### Grounded Criteria (Verified Facts)
Claims that must be verifiable against source documents.

| Criteria Type | What We Check |
|--------------|---------------|
| **Pricing** | Price matches source, within budget |
| **Availability** | Size/stock verified on retailer site |
| **Product Specs** | Materials, dimensions, features accurate |
| **Link Validity** | URL resolves to correct product page |

### Helpfulness Criteria (Value-Add)
Useful information that enhances the recommendation but doesn't require source verification.

- Return policy surfaced
- Fit/sizing guidance provided
- Alternative options offered
- Care instructions included

---

## Grading Process

### Step 1: Response Text Validation
The grader LLM checks if the criterion is satisfied based on what's **explicitly written** in the response.

```
❌ FAIL: Information not stated
❌ FAIL: Information implied but not explicit
✓ PASS: Information clearly stated
```

### Step 2: Source Verification (for Grounded criteria)
If Step 1 passes, claims are verified against scraped source content.

```
❌ FAIL: Claim contradicts source
❌ FAIL: Claim cannot be verified (source unavailable)
✓ PASS: Claim confirmed by source
```

---

## Vertical-Specific Weights

| Vertical | Grounded | Helpfulness | Safety | Completeness |
|----------|----------|-------------|--------|--------------|
| Fashion | 35% | 35% | 15% | 15% |
| Grocery | 35% | 25% | 25% | 15% |
| Electronics | 45% | 25% | 15% | 15% |
| Travel | 40% | 30% | 15% | 15% |
| Home | 40% | 30% | 10% | 20% |

---

## Example Task Breakdown

**Task**: "Find me a black cashmere sweater under $200"

| Criterion | Type | Description | Grounded? |
|-----------|------|-------------|-----------|
| VCI-FASH-001-H | Hurdle | Returns item that is black and cashmere | Yes |
| VCI-FASH-001-1 | Grounded | Price is under $200 (verified) | Yes |
| VCI-FASH-001-2 | Grounded | Size availability is verified | Yes |
| VCI-FASH-001-3 | Grounded | Material/composition is accurate | Yes |
| VCI-FASH-001-4 | Grounded | Link resolves correctly | Yes |
| VCI-FASH-001-5 | Helpfulness | Return policy surfaced | No |
| VCI-FASH-001-6 | Helpfulness | Fit guidance provided | No |
| VCI-FASH-001-7 | Helpfulness | Alternatives offered | No |

**Scoring Example**:
- Hurdle passed: ✓
- Grounded: 4/4 = 100%
- Helpfulness: 2/3 = 67%
- Safety: N/A = 100%
- Completeness: N/A = 100%

**Final Score** = 1.0 × (1.0 × 0.35 + 0.67 × 0.35 + 1.0 × 0.15 + 1.0 × 0.15) = **88.5**

---

## Key Principles

1. **Consumer-Centric**: Scores reflect real purchasing utility
2. **Verifiable**: All grounded claims must be confirmable
3. **Fair Comparison**: All models use the same search infrastructure
4. **Strict but Fair**: No credit for implied or inferred information
5. **Failure is Informative**: Hurdle failures indicate fundamental gaps

---

## Interpreting Scores

| Score Range | Interpretation |
|-------------|----------------|
| 80-100 | Excellent - Ready for consumer use |
| 60-79 | Good - Useful with verification |
| 40-59 | Fair - Significant gaps in accuracy |
| 20-39 | Poor - Major reliability issues |
| 0-19 | Failing - Not suitable for shopping |

---

*For technical implementation details, see the source code in `vci/harness/autograder.py`*
