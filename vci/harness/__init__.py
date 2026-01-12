"""
VCI Harness Module

Three-stage evaluation pipeline:
1. Grounded API Calls - Send prompts to AI models with web search
2. Grounding Pipeline - Scrape sources and map recommendations
3. Autograder - Verify claims and score responses
"""

from .grounded_call import make_grounded_call
from .grounding_pipeline import run_grounding_pipeline
from .autograder import grade_response

__all__ = ["make_grounded_call", "run_grounding_pipeline", "grade_response"]
