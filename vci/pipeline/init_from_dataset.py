#!/usr/bin/env python3
"""
VCI Dataset Initializer

Initializes test cases from CSV dataset files.

Usage:
    python -m vci.pipeline.init_from_dataset --vertical electronics --dataset dev
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

from ..configs.settings import VERTICALS


def parse_criterion_type(type_str: str) -> str:
    """Map CSV criterion type to VCI type."""
    type_lower = type_str.lower()

    if "hurdle" in type_lower:
        return "hurdle"
    elif "safety" in type_lower:
        return "safety"
    elif "helpful" in type_lower:
        return "helpfulness"
    elif "complete" in type_lower:
        return "completeness"
    else:
        return "grounded"


def load_csv_dataset(csv_path: Path) -> List[Dict[str, Any]]:
    """Load and parse a VCI CSV dataset file."""
    tasks = defaultdict(lambda: {
        "criteria": [],
        "prompt": "",
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
                tasks[task_id]["specified_prompt"] = row.get("Specified Prompt", "").strip()
                tasks[task_id]["vertical"] = row.get("Vertical", "").strip().lower()
                tasks[task_id]["workflow"] = row.get("Workflow", "").strip()

            # Add criterion
            criterion = {
                "criterion_id": row.get("Criterion ID", "").strip(),
                "type": parse_criterion_type(row.get("Hurdle Tag", "")),
                "description": row.get("Description", "").strip(),
                "grounded_status": row.get("Criterion Grounding Check", "Grounded").strip(),
                "criteria_type": row.get("Criteria type", "").strip(),
                "product_or_shop": row.get("Shop vs. Product", "Product").strip()
            }

            tasks[task_id]["criteria"].append(criterion)

    # Convert to list format
    result = []
    for task_id, task_data in tasks.items():
        result.append({
            "task_id": task_id,
            "prompt": task_data["prompt"],
            "specified_prompt": task_data.get("specified_prompt", task_data["prompt"]),
            "vertical": task_data["vertical"],
            "workflow": task_data.get("workflow", ""),
            "criteria": task_data["criteria"]
        })

    return result


def save_test_cases(tasks: List[Dict], output_dir: Path):
    """Save test cases as individual JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        task_file = output_dir / f"{task['task_id']}.json"
        with open(task_file, "w") as f:
            json.dump(task, f, indent=2)

    print(f"Saved {len(tasks)} test cases to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Initialize test cases from CSV dataset")
    parser.add_argument("--vertical", required=True, choices=VERTICALS,
                        help="Vertical to initialize")
    parser.add_argument("--dataset", choices=["dev", "eval"], default="dev",
                        help="Dataset to use (default: dev)")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory for test cases")

    args = parser.parse_args()

    # Find CSV file
    csv_filename = f"VCI-{args.vertical.capitalize()}-{args.dataset}.csv"
    csv_path = Path("vci/dataset") / csv_filename

    if not csv_path.exists():
        print(f"Error: Dataset file not found: {csv_path}")
        return 1

    # Load and parse
    print(f"Loading {csv_path}...")
    tasks = load_csv_dataset(csv_path)
    print(f"Found {len(tasks)} tasks with {sum(len(t['criteria']) for t in tasks)} total criteria")

    # Save test cases
    output_dir = args.output_dir or Path("vci/test_cases") / args.vertical / args.dataset
    save_test_cases(tasks, output_dir)

    return 0


if __name__ == "__main__":
    exit(main())
