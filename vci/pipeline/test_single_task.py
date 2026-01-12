#!/usr/bin/env python3
"""
VCI Single Task Tester

Test a single task to validate setup before running full evaluation.

Usage:
    python -m vci.pipeline.test_single_task --task VCI-ELEC-001 --model gpt-4o
"""

import argparse
import json
from pathlib import Path

from ..configs.providers import MODEL_PROVIDER_MAP
from ..harness.grounded_call import TestCase
from .runner import run_task, get_task_dir


def load_test_case_by_id(task_id: str) -> TestCase:
    """Load a test case by its ID."""
    # Search in test_cases directory
    for vertical_dir in Path("vci/test_cases").iterdir():
        if not vertical_dir.is_dir():
            continue

        for dataset_dir in vertical_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            task_file = dataset_dir / f"{task_id}.json"
            if task_file.exists():
                with open(task_file) as f:
                    data = json.load(f)
                return TestCase.from_dict(data)

    raise FileNotFoundError(f"Task not found: {task_id}")


def main():
    parser = argparse.ArgumentParser(description="Test a single VCI task")
    parser.add_argument("--task", required=True,
                        help="Task ID (e.g., VCI-ELEC-001)")
    parser.add_argument("--model", required=True, choices=list(MODEL_PROVIDER_MAP.keys()),
                        help="Model to test with")
    parser.add_argument("--output-dir", type=Path, default=Path("vci/test_output"),
                        help="Output directory for test results")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if results exist")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"VCI Single Task Test")
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")

    # Load test case
    try:
        test_case = load_test_case_by_id(args.task)
        print(f"Loaded task: {test_case.task_id}")
        print(f"Vertical: {test_case.vertical}")
        print(f"Criteria: {len(test_case.criteria)}")
        print(f"\nPrompt:\n{test_case.prompt[:200]}...")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Run task
    task_dir = args.output_dir / args.task
    success = run_task(test_case, args.model, task_dir, force=args.force)

    if success:
        print(f"\n[SUCCESS] Results saved to {task_dir}")

        # Show summary
        results_file = task_dir / "3_autograder_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            print(f"\nScore: {results.get('total_score', 0):.2f}")
            print(f"Hurdle: {'PASS' if results.get('hurdle_passed') else 'FAIL'}")
    else:
        print(f"\n[FAILED] Check logs for errors")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
