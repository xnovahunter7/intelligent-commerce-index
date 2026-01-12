#!/usr/bin/env python3
"""
VCI Full Benchmark Runner

Runs the complete VCI benchmark across all flagship models and verticals.
Supports parallel execution for faster benchmarking.

Usage:
    # Run full benchmark (all models, all verticals) - parallel by default
    python -m vci.pipeline.run_benchmark

    # Run specific models
    python -m vci.pipeline.run_benchmark --models gpt-5 opus-4.5 gemini-3-pro

    # Run specific verticals
    python -m vci.pipeline.run_benchmark --verticals electronics fashion

    # Run sequentially (disable parallel)
    python -m vci.pipeline.run_benchmark --sequential

    # Control parallelism (default: 12 workers, one per model)
    python -m vci.pipeline.run_benchmark --workers 6

    # Dry run (show what would be run)
    python -m vci.pipeline.run_benchmark --dry-run
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Optional

# Load .env file
def load_env_file():
    """Load .env file manually without python-dotenv dependency."""
    env_paths = [
        Path(__file__).parent.parent.parent / ".env",  # vice/.env
        Path.cwd() / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
            break

load_env_file()

from ..configs.settings import SETTINGS, VERTICALS
from ..configs.providers import VCI_BENCHMARK_MODELS, MODEL_REGISTRY, get_openrouter_model_id
from ..utils.cost_tracker import get_global_tracker, reset_global_tracker
from .runner import run_all_tasks, load_tasks_for_vertical


# Global lock for thread-safe printing
print_lock = Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)


def run_model_benchmark(
    model: str,
    verticals: List[str],
    run: int,
    dataset: str,
    force: bool,
    limit: Optional[int],
) -> dict:
    """
    Run benchmark for a single model across all verticals.
    This function is designed to be run in a thread.
    """
    model_start = time.time()
    safe_print(f"\n🚀 [{model}] Starting benchmark...")

    model_results = []
    completed_count = 0
    failed_count = 0

    for vertical in verticals:
        try:
            result = run_all_tasks(
                vertical=vertical,
                model=model,
                run=run,
                dataset=dataset,
                force=force,
                limit=limit
            )
            result["model"] = model
            result["vertical"] = vertical
            model_results.append(result)
            completed_count += result.get("completed", 0)
            failed_count += result.get("failed", 0)
            safe_print(f"  ✓ [{model}] {vertical}: {result.get('completed', 0)} completed, {result.get('failed', 0)} failed")
        except Exception as e:
            safe_print(f"  ✗ [{model}] {vertical}: ERROR - {e}")
            model_results.append({
                "model": model,
                "vertical": vertical,
                "error": str(e)
            })
            failed_count += 1

    model_time = time.time() - model_start
    safe_print(f"✅ [{model}] Complete in {model_time:.1f}s - {completed_count} completed, {failed_count} failed")

    return {
        "model": model,
        "results": model_results,
        "completed": completed_count,
        "failed": failed_count,
        "time_seconds": model_time
    }


def run_full_benchmark(
    models: List[str],
    verticals: List[str],
    run: int = 1,
    dataset: str = "dev",
    force: bool = False,
    dry_run: bool = False,
    limit: Optional[int] = None,
    parallel: bool = True,
    workers: Optional[int] = None,
) -> dict:
    """
    Run the full VCI benchmark.

    Args:
        models: List of model names to evaluate
        verticals: List of verticals to evaluate
        run: Run number
        dataset: Dataset to use (dev or eval)
        force: Force re-run of completed tasks
        dry_run: Just print what would be run
        limit: Limit tasks per vertical
        parallel: Run models in parallel (default True)
        workers: Number of parallel workers (default: number of models)

    Returns:
        Summary of results
    """
    total_combinations = len(models) * len(verticals)
    num_workers = workers or len(models)

    print(f"\n{'='*70}")
    print(f"VCI FULL BENCHMARK")
    print(f"{'='*70}")
    print(f"Models: {len(models)}")
    for m in models:
        print(f"  - {m}")
    print(f"Verticals: {len(verticals)}")
    for v in verticals:
        print(f"  - {v}")
    print(f"Total combinations: {total_combinations}")
    print(f"Dataset: {dataset}")
    print(f"Run: {run}")
    print(f"Parallel: {parallel} (workers: {num_workers})")
    print(f"{'='*70}\n")

    if dry_run:
        print("DRY RUN - No tasks will be executed\n")
        for model in models:
            for vertical in verticals:
                tasks = load_tasks_for_vertical(vertical, dataset)
                task_count = len(tasks) if not limit else min(len(tasks), limit)
                print(f"  {model} x {vertical}: {task_count} tasks")
        print(f"\nTotal tasks: {sum(len(load_tasks_for_vertical(v, dataset)) for v in verticals) * len(models)}")
        return {"dry_run": True}

    # Initialize cost tracking
    reset_global_tracker(run_id=f"run_{run}_{dataset}")

    start_time = time.time()
    all_results = []

    if parallel and len(models) > 1:
        # Run models in parallel using ThreadPoolExecutor
        print(f"🔄 Running {len(models)} models in parallel with {num_workers} workers...\n")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all model benchmarks
            future_to_model = {
                executor.submit(
                    run_model_benchmark,
                    model, verticals, run, dataset, force, limit
                ): model
                for model in models
            }

            # Collect results as they complete
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    model_result = future.result()
                    all_results.extend(model_result["results"])
                except Exception as e:
                    safe_print(f"❌ [{model}] Failed with exception: {e}")
                    all_results.append({
                        "model": model,
                        "error": str(e)
                    })
    else:
        # Run sequentially
        print("Running models sequentially...\n")
        for i, model in enumerate(models):
            for j, vertical in enumerate(verticals):
                combo_num = i * len(verticals) + j + 1
                print(f"\n[{combo_num}/{total_combinations}] {model} x {vertical}")
                print(f"{'-'*50}")

                try:
                    result = run_all_tasks(
                        vertical=vertical,
                        model=model,
                        run=run,
                        dataset=dataset,
                        force=force,
                        limit=limit
                    )
                    result["model"] = model
                    result["vertical"] = vertical
                    all_results.append(result)
                except Exception as e:
                    print(f"ERROR: {e}")
                    all_results.append({
                        "model": model,
                        "vertical": vertical,
                        "error": str(e)
                    })

    total_time = time.time() - start_time

    # Summary
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Speedup from parallel: ~{len(models)}x" if parallel else "")
    print(f"\nResults by model:")

    for model in models:
        model_results = [r for r in all_results if r.get("model") == model]
        completed = sum(r.get("completed", 0) for r in model_results)
        failed = sum(r.get("failed", 0) for r in model_results)
        errors = sum(1 for r in model_results if "error" in r)
        print(f"  {model}: {completed} completed, {failed} failed, {errors} errors")

    # Save benchmark summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "verticals": verticals,
        "run": run,
        "dataset": dataset,
        "parallel": parallel,
        "workers": num_workers,
        "total_time_seconds": total_time,
        "results": all_results
    }

    summary_path = Path(SETTINGS.results_dir) / f"benchmark_run_{run}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    # Print cost summary
    cost_tracker = get_global_tracker()
    cost_tracker.print_summary()

    # Save cost report
    cost_report_path = cost_tracker.save_report(Path(SETTINGS.results_dir))
    print(f"Cost report saved to: {cost_report_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run the full VCI benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark (all 12 flagship models, all 5 verticals)
  python -m vci.pipeline.run_benchmark

  # Run specific models
  python -m vci.pipeline.run_benchmark --models gpt-5 opus-4.5

  # Run specific verticals
  python -m vci.pipeline.run_benchmark --verticals electronics fashion

  # Dry run to see what would be executed
  python -m vci.pipeline.run_benchmark --dry-run

  # Limit tasks per vertical (for testing)
  python -m vci.pipeline.run_benchmark --limit 1
        """
    )

    parser.add_argument(
        "--models", nargs="+",
        default=VCI_BENCHMARK_MODELS,
        help=f"Models to evaluate (default: all {len(VCI_BENCHMARK_MODELS)} flagship models)"
    )
    parser.add_argument(
        "--verticals", nargs="+",
        default=VERTICALS,
        choices=VERTICALS,
        help=f"Verticals to evaluate (default: all {len(VERTICALS)})"
    )
    parser.add_argument(
        "--run", type=int, default=1,
        help="Run number (default: 1)"
    )
    parser.add_argument(
        "--dataset", choices=["dev", "eval"], default="dev",
        help="Dataset to use (default: dev)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run of completed tasks"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be run without executing"
    )
    parser.add_argument(
        "--limit", type=int,
        help="Limit number of tasks per vertical (for testing)"
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run models sequentially instead of in parallel"
    )
    parser.add_argument(
        "--workers", type=int,
        help="Number of parallel workers (default: number of models)"
    )

    args = parser.parse_args()

    # Validate models
    invalid_models = []
    for model in args.models:
        try:
            get_openrouter_model_id(model)
        except ValueError:
            invalid_models.append(model)

    if invalid_models:
        print(f"Error: Unknown models: {invalid_models}")
        print(f"Available models: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set")
        print("Set it in .env file or environment variable")
        sys.exit(1)

    run_full_benchmark(
        models=args.models,
        verticals=args.verticals,
        run=args.run,
        dataset=args.dataset,
        force=args.force,
        dry_run=args.dry_run,
        limit=args.limit,
        parallel=not args.sequential,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
