#!/usr/bin/env python3
"""
VCI Run All Models

Launches evaluation for all configured models in parallel.

Usage:
    python -m vci.pipeline.run_all_models --vertical electronics --run 1
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

from ..configs.settings import VERTICALS
from ..configs.providers import MODEL_REGISTRY


# Default models to evaluate
DEFAULT_MODELS = [
    "gpt-4o",
    "claude-3.5-sonnet",
    "gemini-2.0-flash",
    "sonar",
]


def run_model_evaluation(
    model: str,
    vertical: str,
    run: int,
    dataset: str
) -> dict:
    """Run evaluation for a single model (in subprocess)."""
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "vci.pipeline.runner",
                "--vertical", vertical,
                "--model", model,
                "--run", str(run),
                "--dataset", dataset
            ],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per model
        )

        return {
            "model": model,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "model": model,
            "success": False,
            "error": "Timeout after 1 hour"
        }
    except Exception as e:
        return {
            "model": model,
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Run VCI evaluation for all models")
    parser.add_argument("--vertical", required=True, choices=VERTICALS,
                        help="Vertical to evaluate")
    parser.add_argument("--run", type=int, default=1,
                        help="Run number (default: 1)")
    parser.add_argument("--dataset", choices=["dev", "eval"], default="dev",
                        help="Dataset to use (default: dev)")
    parser.add_argument("--models", nargs="+",
                        default=DEFAULT_MODELS,
                        help="Models to evaluate (default: all)")
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="Maximum parallel model evaluations (default: 4)")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"VCI Evaluation: {args.vertical} - All Models")
    print(f"Models: {', '.join(args.models)}")
    print(f"{'='*60}\n")

    results = []

    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(
                run_model_evaluation,
                model,
                args.vertical,
                args.run,
                args.dataset
            ): model
            for model in args.models
        }

        for future in as_completed(futures):
            model = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "SUCCESS" if result["success"] else "FAILED"
                print(f"[{status}] {model}")
            except Exception as e:
                print(f"[ERROR] {model}: {e}")
                results.append({"model": model, "success": False, "error": str(e)})

    # Summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"Complete: {successful}/{len(results)} models succeeded")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
