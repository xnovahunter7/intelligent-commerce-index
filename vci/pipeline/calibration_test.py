#!/usr/bin/env python3
"""
VCI Grader Calibration Test

Tests grader consistency by running the same response through the grader multiple times.
Measures variance to detect unreliable grading.

Usage:
    python -m vci.pipeline.calibration_test --task VCI-ELEC-001 --model gpt-4o --runs 3
    python -m vci.pipeline.calibration_test --all-tasks --runs 3
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..configs.settings import SETTINGS, VERTICALS
from ..harness.autograder import grade_response
from .runner import load_tasks_for_vertical, get_results_dir


@dataclass
class CalibrationResult:
    """Result of calibration test for a single criterion."""
    criterion_id: str
    scores: List[float]
    mean: float
    std_dev: float
    is_consistent: bool  # True if std_dev < threshold

    def to_dict(self) -> dict:
        return {
            "criterion_id": self.criterion_id,
            "scores": self.scores,
            "mean": round(self.mean, 3),
            "std_dev": round(self.std_dev, 3),
            "is_consistent": self.is_consistent
        }


@dataclass
class TaskCalibration:
    """Calibration results for a full task."""
    task_id: str
    total_scores: List[float]
    total_mean: float
    total_std_dev: float
    criterion_results: List[CalibrationResult]
    consistency_rate: float  # % of criteria that are consistent

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "total_scores": self.total_scores,
            "total_mean": round(self.total_mean, 3),
            "total_std_dev": round(self.total_std_dev, 3),
            "criterion_results": [c.to_dict() for c in self.criterion_results],
            "consistency_rate": round(self.consistency_rate, 1)
        }


def run_calibration_test(
    task_id: str,
    model: str,
    run: int,
    num_runs: int = 3,
    consistency_threshold: float = 0.1
) -> Optional[TaskCalibration]:
    """
    Run calibration test for a single task.

    Args:
        task_id: Task to test
        model: Model that generated the response
        run: Run number to use
        num_runs: Number of grading runs
        consistency_threshold: Max std_dev for criterion to be "consistent"

    Returns:
        TaskCalibration with results, or None if task not found
    """
    # Find the task results
    results_dir = Path(SETTINGS.results_dir)
    model_dir = model.replace("/", "_")

    # Search for the task across verticals
    task_dir = None
    vertical = None

    for v in VERTICALS:
        potential_dir = results_dir / model_dir / v / f"run_{run}" / f"task_{task_id}"
        if potential_dir.exists():
            task_dir = potential_dir
            vertical = v
            break

    if not task_dir:
        print(f"Task {task_id} not found for model {model} run {run}")
        return None

    # Load the grounded response and grounding result
    grounded_file = task_dir / "1_grounded_response.json"
    grounding_file = task_dir / "2_scraped_sources.json"
    test_case_file = task_dir / "0_test_case.json"

    if not grounded_file.exists():
        print(f"No grounded response found for {task_id}")
        return None

    with open(grounded_file) as f:
        grounded_response = json.load(f)

    # Load grounding result or create empty one
    if grounding_file.exists():
        with open(grounding_file) as f:
            grounding_result = json.load(f)
    else:
        grounding_result = {
            "recommendations": [],
            "scraped_sources": [],
            "product_source_map": [],
            "failed_scrapes": []
        }

    # Load criteria from test case
    if test_case_file.exists():
        with open(test_case_file) as f:
            test_case = json.load(f)
            criteria = test_case.get("criteria", [])
    else:
        print(f"No test case found for {task_id}")
        return None

    # Run grading multiple times
    all_results = []
    print(f"\nRunning {num_runs} grading iterations for {task_id}...")

    for i in range(num_runs):
        print(f"  Iteration {i+1}/{num_runs}...", end=" ", flush=True)

        result = grade_response(
            grounded_response,
            grounding_result,
            criteria,
            vertical
        )

        all_results.append(result)
        print(f"Score: {result.total_score:.3f}")

    # Analyze results
    total_scores = [r.total_score for r in all_results]
    total_mean = statistics.mean(total_scores)
    total_std = statistics.stdev(total_scores) if len(total_scores) > 1 else 0.0

    # Per-criterion analysis
    criterion_results = []
    criterion_ids = set()
    for r in all_results:
        for cr in r.criterion_results:
            criterion_ids.add(cr.criterion_id)

    consistent_count = 0
    for cid in sorted(criterion_ids):
        scores = []
        for r in all_results:
            for cr in r.criterion_results:
                if cr.criterion_id == cid:
                    scores.append(cr.score)
                    break

        if scores:
            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            is_consistent = std <= consistency_threshold
            if is_consistent:
                consistent_count += 1

            criterion_results.append(CalibrationResult(
                criterion_id=cid,
                scores=scores,
                mean=mean,
                std_dev=std,
                is_consistent=is_consistent
            ))

    consistency_rate = (consistent_count / len(criterion_results) * 100) if criterion_results else 100.0

    return TaskCalibration(
        task_id=task_id,
        total_scores=total_scores,
        total_mean=total_mean,
        total_std_dev=total_std,
        criterion_results=criterion_results,
        consistency_rate=consistency_rate
    )


def run_all_calibration(
    model: str,
    run: int,
    num_runs: int = 3,
    limit: Optional[int] = None
) -> List[TaskCalibration]:
    """Run calibration test on all available tasks."""
    results_dir = Path(SETTINGS.results_dir)
    model_dir = model.replace("/", "_")

    all_calibrations = []
    task_count = 0

    for vertical in VERTICALS:
        run_dir = results_dir / model_dir / vertical / f"run_{run}"
        if not run_dir.exists():
            continue

        for task_dir in sorted(run_dir.iterdir()):
            if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
                continue

            task_id = task_dir.name.replace("task_", "")

            if limit and task_count >= limit:
                break

            calibration = run_calibration_test(task_id, model, run, num_runs)
            if calibration:
                all_calibrations.append(calibration)
                task_count += 1

        if limit and task_count >= limit:
            break

    return all_calibrations


def print_calibration_report(calibrations: List[TaskCalibration]):
    """Print a summary report of calibration results."""
    if not calibrations:
        print("No calibration results to report.")
        return

    print("\n" + "=" * 70)
    print("GRADER CALIBRATION REPORT")
    print("=" * 70)

    # Overall statistics
    all_std_devs = [c.total_std_dev for c in calibrations]
    all_consistency_rates = [c.consistency_rate for c in calibrations]

    print(f"\nTasks tested: {len(calibrations)}")
    print(f"Average score std_dev: {statistics.mean(all_std_devs):.4f}")
    print(f"Max score std_dev: {max(all_std_devs):.4f}")
    print(f"Average criterion consistency: {statistics.mean(all_consistency_rates):.1f}%")

    # Per-task breakdown
    print("\n" + "-" * 70)
    print(f"{'Task ID':<20} {'Mean Score':<12} {'Std Dev':<10} {'Consistency':<12}")
    print("-" * 70)

    for cal in calibrations:
        status = "OK" if cal.total_std_dev < 0.05 else "WARN" if cal.total_std_dev < 0.1 else "HIGH"
        print(f"{cal.task_id:<20} {cal.total_mean:<12.3f} {cal.total_std_dev:<10.4f} {cal.consistency_rate:<10.1f}% [{status}]")

    # Flag inconsistent criteria
    inconsistent = []
    for cal in calibrations:
        for cr in cal.criterion_results:
            if not cr.is_consistent:
                inconsistent.append((cal.task_id, cr.criterion_id, cr.std_dev))

    if inconsistent:
        print("\n" + "-" * 70)
        print("INCONSISTENT CRITERIA (std_dev > 0.1):")
        print("-" * 70)
        for task_id, crit_id, std in sorted(inconsistent, key=lambda x: -x[2])[:10]:
            print(f"  {task_id}/{crit_id}: std_dev = {std:.4f}")

    # Summary verdict
    avg_consistency = statistics.mean(all_consistency_rates)
    avg_std = statistics.mean(all_std_devs)

    print("\n" + "=" * 70)
    if avg_consistency >= 90 and avg_std < 0.05:
        print("VERDICT: GRADER IS RELIABLE")
        print("High consistency and low variance across runs.")
    elif avg_consistency >= 75 and avg_std < 0.1:
        print("VERDICT: GRADER IS ACCEPTABLE")
        print("Moderate consistency. Some criteria show variance.")
    else:
        print("VERDICT: GRADER NEEDS IMPROVEMENT")
        print("High variance detected. Consider:")
        print("  - Making grading prompts more specific")
        print("  - Adding example-based anchoring")
        print("  - Using deterministic grading where possible")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test grader calibration/consistency")
    parser.add_argument("--task", help="Specific task ID to test")
    parser.add_argument("--model", default="gpt-4o", help="Model to test against")
    parser.add_argument("--run", type=int, default=1, help="Run number")
    parser.add_argument("--runs", type=int, default=3, help="Number of grading iterations")
    parser.add_argument("--all-tasks", action="store_true", help="Test all available tasks")
    parser.add_argument("--limit", type=int, help="Limit number of tasks when using --all-tasks")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")

    args = parser.parse_args()

    if args.task:
        calibration = run_calibration_test(args.task, args.model, args.run, args.runs)
        if calibration:
            print_calibration_report([calibration])
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(calibration.to_dict(), f, indent=2)
    elif args.all_tasks:
        calibrations = run_all_calibration(args.model, args.run, args.runs, args.limit)
        print_calibration_report(calibrations)
        if args.output:
            with open(args.output, "w") as f:
                json.dump([c.to_dict() for c in calibrations], f, indent=2)
    else:
        print("Please specify --task or --all-tasks")
        sys.exit(1)


if __name__ == "__main__":
    main()
