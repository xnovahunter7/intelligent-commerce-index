#!/usr/bin/env python3
"""
VCI Results Exporter

Exports evaluation results to CSV for analysis.

Usage:
    python -m vci.pipeline.export_results --output results_summary.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..configs.settings import SETTINGS, VERTICALS
from ..configs.providers import MODEL_PROVIDER_MAP


def collect_all_results(results_base: Path) -> List[Dict[str, Any]]:
    """Collect all completed evaluation results."""
    results = []

    for provider_dir in results_base.iterdir():
        if not provider_dir.is_dir():
            continue

        for model_dir in provider_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for vertical_dir in model_dir.iterdir():
                if not vertical_dir.is_dir():
                    continue

                for run_dir in vertical_dir.iterdir():
                    if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                        continue

                    # Find all completed tasks
                    for task_dir in run_dir.iterdir():
                        if not task_dir.is_dir():
                            continue

                        results_file = task_dir / "3_autograder_results.json"
                        if not results_file.exists():
                            continue

                        with open(results_file) as f:
                            task_result = json.load(f)

                        results.append({
                            "provider": provider_dir.name,
                            "model": model_dir.name,
                            "vertical": vertical_dir.name,
                            "run": run_dir.name.replace("run_", ""),
                            "task_id": task_result.get("task_id", task_dir.name),
                            "hurdle_passed": task_result.get("hurdle_passed", False),
                            "total_score": task_result.get("total_score", 0),
                            "grounded_score": task_result.get("component_scores", {}).get("grounded", 0),
                            "helpfulness_score": task_result.get("component_scores", {}).get("helpfulness", 0),
                            "safety_score": task_result.get("component_scores", {}).get("safety", 0),
                            "completeness_score": task_result.get("component_scores", {}).get("completeness", 0),
                            "num_criteria": len(task_result.get("criterion_results", [])),
                            "latency_ms": task_result.get("total_latency_ms", 0)
                        })

    return results


def aggregate_by_model(results: List[Dict]) -> List[Dict]:
    """Aggregate results by model for leaderboard."""
    from collections import defaultdict

    model_stats = defaultdict(lambda: {
        "total_tasks": 0,
        "hurdles_passed": 0,
        "total_score": 0,
        "grounded_total": 0,
        "helpfulness_total": 0,
        "safety_total": 0,
        "completeness_total": 0
    })

    for r in results:
        key = (r["provider"], r["model"])
        stats = model_stats[key]
        stats["total_tasks"] += 1
        stats["hurdles_passed"] += 1 if r["hurdle_passed"] else 0
        stats["total_score"] += r["total_score"]
        stats["grounded_total"] += r["grounded_score"]
        stats["helpfulness_total"] += r["helpfulness_score"]
        stats["safety_total"] += r["safety_score"]
        stats["completeness_total"] += r["completeness_score"]

    aggregated = []
    for (provider, model), stats in model_stats.items():
        n = stats["total_tasks"]
        if n == 0:
            continue

        aggregated.append({
            "provider": provider,
            "model": model,
            "total_tasks": n,
            "hurdle_pass_rate": stats["hurdles_passed"] / n,
            "avg_score": stats["total_score"] / n,
            "avg_grounded": stats["grounded_total"] / n,
            "avg_helpfulness": stats["helpfulness_total"] / n,
            "avg_safety": stats["safety_total"] / n,
            "avg_completeness": stats["completeness_total"] / n
        })

    # Sort by average score descending
    aggregated.sort(key=lambda x: x["avg_score"], reverse=True)

    return aggregated


def export_to_csv(data: List[Dict], output_path: Path, fieldnames: Optional[List[str]] = None):
    """Export data to CSV file."""
    if not data:
        print("No data to export")
        return

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Exported {len(data)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export VCI results to CSV")
    parser.add_argument("--output", type=Path, default=Path("results_summary.csv"),
                        help="Output CSV file path")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate results by model for leaderboard")
    parser.add_argument("--results-dir", type=Path, default=Path(SETTINGS.results_dir),
                        help="Results directory to scan")

    args = parser.parse_args()

    print(f"Scanning results in {args.results_dir}...")
    results = collect_all_results(args.results_dir)
    print(f"Found {len(results)} completed task evaluations")

    if args.aggregate:
        results = aggregate_by_model(results)
        print(f"Aggregated to {len(results)} model summaries")

    export_to_csv(results, args.output)


if __name__ == "__main__":
    main()
