#!/usr/bin/env python3
"""
VCI Task Runner

Executes all tasks for a given vertical/model/run combination.

Usage:
    python -m vci.pipeline.runner --vertical electronics --model gpt-4o --run 1
    python -m vci.pipeline.runner --vertical fashion --model claude-3.5-sonnet --run 1
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..configs.settings import SETTINGS, VERTICALS
from ..configs.providers import MODEL_REGISTRY, get_openrouter_model_id
from ..harness.grounded_call import make_grounded_call, TestCase
from ..harness.grounding_pipeline import run_grounding_pipeline
from ..harness.autograder import grade_response


def get_results_dir(model: str, vertical: str, run: int) -> Path:
    """Get the results directory for a specific run."""
    # Normalize model name for directory
    model_dir = model.replace("/", "_")
    return Path(SETTINGS.results_dir) / model_dir / vertical / f"run_{run}"


def get_task_dir(results_dir: Path, task_id: str) -> Path:
    """Get the directory for a specific task's results."""
    return results_dir / f"task_{task_id}"


def is_task_complete(task_dir: Path) -> bool:
    """Check if a task has been fully evaluated (all 3 stages complete)."""
    return (task_dir / "3_autograder_results.json").exists()


def load_tasks_from_csv(csv_path: Path) -> List[TestCase]:
    """Load and group tasks from a VCI CSV dataset file."""
    if not csv_path.exists():
        return []

    tasks = defaultdict(lambda: {
        "criteria": [],
        "prompt": "",
        "specified_prompt": "",
        "vertical": ""
    })

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            task_id = row.get("Task ID", "").strip()
            if not task_id:
                continue

            # First row for this task sets the prompt
            if not tasks[task_id]["prompt"]:
                tasks[task_id]["prompt"] = row.get("Prompt", "").strip()
                tasks[task_id]["specified_prompt"] = row.get("Specified Prompt", "").strip() or row.get("Prompt", "").strip()
                tasks[task_id]["vertical"] = row.get("Vertical", "").strip().lower()

            # Add criterion
            criterion = {
                "criterion_id": row.get("Criterion ID", "").strip(),
                "type": "hurdle" if row.get("Hurdle Tag", "").strip().lower() == "hurdle" else "grounded",
                "description": row.get("Description", "").strip(),
                "grounded_status": row.get("Criterion Grounding Check", "Grounded").strip(),
                "criteria_type": row.get("Criteria type", "").strip(),
            }
            tasks[task_id]["criteria"].append(criterion)

    # Convert to TestCase objects
    result = []
    for task_id, task_data in tasks.items():
        result.append(TestCase(
            task_id=task_id,
            prompt=task_data["prompt"],
            specified_prompt=task_data["specified_prompt"],
            vertical=task_data["vertical"],
            criteria=task_data["criteria"]
        ))

    return result


def load_tasks_for_vertical(vertical: str, dataset: str = "dev") -> List[TestCase]:
    """Load all tasks for a vertical from the dataset."""
    # Try different path patterns
    possible_paths = [
        Path(f"vci/dataset/VCI-{vertical.capitalize()}-{dataset}.csv"),
        Path(f"dataset/VCI-{vertical.capitalize()}-{dataset}.csv"),
        Path(__file__).parent.parent / "dataset" / f"VCI-{vertical.capitalize()}-{dataset}.csv",
    ]

    for csv_path in possible_paths:
        if csv_path.exists():
            return load_tasks_from_csv(csv_path)

    print(f"Warning: Dataset not found. Tried: {possible_paths}")
    return []


def run_task(
    test_case: TestCase,
    model: str,
    task_dir: Path,
    force: bool = False
) -> bool:
    """
    Run a single task through all 3 stages of the pipeline.

    Stage 1: Grounded call (get model response)
    Stage 2: Grounding pipeline (scrape sources)
    Stage 3: Autograder (verify claims and score)

    Returns True if successful, False otherwise.
    """
    task_dir.mkdir(parents=True, exist_ok=True)

    # Stage 0: Save test case
    test_case_path = task_dir / "0_test_case.json"
    if not test_case_path.exists() or force:
        with open(test_case_path, "w") as f:
            json.dump(test_case.to_dict(), f, indent=2)

    # Stage 1: Grounded call
    grounded_path = task_dir / "1_grounded_response.json"
    grounded_response_dict = None

    if not grounded_path.exists() or force:
        print(f"  Stage 1: Making grounded call to {model}...")
        try:
            grounded_response = make_grounded_call(test_case, model, grounded_path)
            print(f"    Response: {len(grounded_response.response_text)} chars, {len(grounded_response.grounding_chunks)} URLs found")
            print(f"    Latency: {grounded_response.latency_ms:.0f}ms")
            with open(grounded_path) as f:
                grounded_response_dict = json.load(f)
        except Exception as e:
            print(f"    Error: {e}")
            return False
    else:
        print(f"  Stage 1: Using cached response")
        with open(grounded_path) as f:
            grounded_response_dict = json.load(f)

    # Stage 2: Grounding pipeline (scrape sources)
    grounding_path = task_dir / "2_scraped_sources.json"
    grounding_result_dict = None

    if not grounding_path.exists() or force:
        print(f"  Stage 2: Running grounding pipeline...")
        try:
            grounding_result = run_grounding_pipeline(
                grounded_response_dict,
                test_case.vertical,
                grounding_path
            )
            print(f"    Sources scraped: {len(grounding_result.scraped_sources)}, Failed: {len(grounding_result.failed_scrapes)}")
            print(f"    Recommendations found: {len(grounding_result.recommendations)}")
            with open(grounding_path) as f:
                grounding_result_dict = json.load(f)
        except Exception as e:
            print(f"    Error in grounding: {e}")
            # Create empty grounding result to continue
            grounding_result_dict = {
                "task_id": test_case.task_id,
                "recommendations": [],
                "scraped_sources": [],
                "product_source_map": [],
                "failed_scrapes": [],
                "timestamp": 0,
                "total_latency_ms": 0
            }
            with open(grounding_path, "w") as f:
                json.dump(grounding_result_dict, f, indent=2)
    else:
        print(f"  Stage 2: Using cached grounding")
        with open(grounding_path) as f:
            grounding_result_dict = json.load(f)

    # Stage 3: Autograder
    autograder_path = task_dir / "3_autograder_results.json"

    if not autograder_path.exists() or force:
        print(f"  Stage 3: Running autograder...")
        try:
            task_result = grade_response(
                grounded_response_dict,
                grounding_result_dict,
                test_case.criteria,
                test_case.vertical,
                autograder_path
            )
            print(f"    Hurdle passed: {task_result.hurdle_passed}")
            print(f"    Total score: {task_result.total_score:.2f}")
            print(f"    Component scores: {task_result.component_scores}")
            return True
        except Exception as e:
            print(f"    Error in autograder: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"  Stage 3: Using cached autograder results")
        return True


def run_all_tasks(
    vertical: str,
    model: str,
    run: int,
    dataset: str = "dev",
    force: bool = False,
    limit: Optional[int] = None
) -> dict:
    """
    Run all tasks for a vertical/model combination.

    Returns summary statistics.
    """
    print(f"\n{'='*60}")
    print(f"VCI Evaluation: {vertical} / {model} / run_{run}")
    print(f"{'='*60}\n")

    results_dir = get_results_dir(model, vertical, run)
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks_for_vertical(vertical, dataset)
    if limit:
        tasks = tasks[:limit]

    if not tasks:
        print("No tasks found!")
        return {"total": 0, "completed": 0, "failed": 0, "skipped": 0}

    print(f"Found {len(tasks)} tasks\n")

    completed = 0
    failed = 0
    skipped = 0

    for i, task in enumerate(tasks):
        task_dir = get_task_dir(results_dir, task.task_id)

        if is_task_complete(task_dir) and not force:
            print(f"[{i+1}/{len(tasks)}] Task {task.task_id}: SKIPPED (already complete)")
            skipped += 1
            continue

        print(f"[{i+1}/{len(tasks)}] Task {task.task_id}:")
        print(f"  Prompt: {task.prompt[:60]}...")

        if run_task(task, model, task_dir, force):
            completed += 1
        else:
            failed += 1

        print()

    summary = {
        "total": len(tasks),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "model": model,
        "vertical": vertical,
        "run": run
    }

    print(f"\n{'='*60}")
    print(f"Summary: {completed} completed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}\n")

    # Save run summary
    with open(results_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run VCI evaluation tasks")
    parser.add_argument("--vertical", required=True, choices=VERTICALS,
                        help="Vertical to evaluate")
    parser.add_argument("--model", required=True,
                        help=f"Model to evaluate. Short names: {list(MODEL_REGISTRY.keys())[:5]}... or full OpenRouter ID")
    parser.add_argument("--run", type=int, default=1,
                        help="Run number (default: 1)")
    parser.add_argument("--dataset", choices=["dev", "eval"], default="dev",
                        help="Dataset to use (default: dev)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run of completed tasks")
    parser.add_argument("--limit", type=int,
                        help="Limit number of tasks to run")

    args = parser.parse_args()

    # Validate model
    try:
        model_id = get_openrouter_model_id(args.model)
        print(f"Using model: {model_id}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    run_all_tasks(
        vertical=args.vertical,
        model=args.model,
        run=args.run,
        dataset=args.dataset,
        force=args.force,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
