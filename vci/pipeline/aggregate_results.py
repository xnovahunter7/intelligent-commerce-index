#!/usr/bin/env python3
"""
VCI Results Aggregation

Aggregates benchmark results across models and verticals to produce:
- VCI scores per model
- Leaderboard rankings
- Detailed breakdowns by vertical and criteria type

Usage:
    python -m vci.pipeline.aggregate_results --run 1
    python -m vci.pipeline.aggregate_results --run 1 --format markdown
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..configs.settings import SETTINGS, VERTICALS, VERTICAL_WEIGHTS


@dataclass
class ModelScore:
    """Aggregated score for a single model."""
    model: str
    vci_score: float  # Overall VCI score (0-100)
    vertical_scores: Dict[str, float]  # Score per vertical
    hurdle_pass_rate: float  # % of tasks that passed hurdle
    grounded_score: float  # Average grounded component score
    helpfulness_score: float  # Average helpfulness component score
    tasks_completed: int
    tasks_failed: int
    avg_latency_ms: float


@dataclass
class LeaderboardEntry:
    """Entry in the VCI leaderboard."""
    rank: int
    model: str
    vci_score: float
    vertical_scores: Dict[str, float]
    hurdle_pass_rate: float


def load_task_results(results_dir: Path, model: str, vertical: str, run: int) -> List[Dict]:
    """Load all task results for a model/vertical combination."""
    model_dir = model.replace("/", "_")
    run_dir = results_dir / model_dir / vertical / f"run_{run}"

    results = []
    if not run_dir.exists():
        return results

    for task_dir in run_dir.iterdir():
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue

        # Load grounded response (Stage 1)
        grounded_file = task_dir / "1_grounded_response.json"
        if grounded_file.exists():
            with open(grounded_file) as f:
                grounded_response = json.load(f)
                results.append({
                    "task_id": task_dir.name.replace("task_", ""),
                    "model": model,
                    "vertical": vertical,
                    "grounded_response": grounded_response,
                })

    return results


def calculate_task_score(grounded_response: Dict) -> Dict[str, Any]:
    """
    Calculate score for a single task from grounded response.

    Uses the criteria stored in the response to calculate:
    - hurdle_passed: Did all hurdle criteria pass?
    - component_scores: Scores by criteria type
    - total_score: Weighted average
    """
    criteria = grounded_response.get("criteria", [])

    if not criteria:
        return {
            "hurdle_passed": False,
            "total_score": 0.0,
            "component_scores": {},
            "latency_ms": grounded_response.get("latency_ms", 0)
        }

    # Check hurdle criteria
    hurdle_criteria = [c for c in criteria if c.get("type") == "hurdle"]
    hurdle_passed = all(c.get("grounded_status") == "Grounded" for c in hurdle_criteria)

    if not hurdle_passed:
        return {
            "hurdle_passed": False,
            "total_score": 0.0,
            "component_scores": {"grounded": 0, "helpfulness": 0},
            "latency_ms": grounded_response.get("latency_ms", 0)
        }

    # Calculate component scores
    grounded_criteria = [c for c in criteria if c.get("type") == "grounded"]
    grounded_passed = sum(1 for c in grounded_criteria if c.get("grounded_status") == "Grounded")
    grounded_score = grounded_passed / len(grounded_criteria) if grounded_criteria else 1.0

    # Helpfulness criteria (those marked as "Not Grounded" but should be checked)
    helpfulness_criteria = [c for c in criteria if c.get("criteria_type") == "Helpfulness"]
    helpfulness_passed = sum(1 for c in helpfulness_criteria if c.get("grounded_status") == "Grounded")
    helpfulness_score = helpfulness_passed / len(helpfulness_criteria) if helpfulness_criteria else 1.0

    component_scores = {
        "grounded": grounded_score,
        "helpfulness": helpfulness_score,
    }

    # Simple weighted average (40% grounded, 30% helpfulness, 30% other)
    total_score = (grounded_score * 0.5 + helpfulness_score * 0.5)

    return {
        "hurdle_passed": hurdle_passed,
        "total_score": total_score,
        "component_scores": component_scores,
        "latency_ms": grounded_response.get("latency_ms", 0)
    }


def aggregate_model_scores(
    results_dir: Path,
    models: List[str],
    verticals: List[str],
    run: int
) -> List[ModelScore]:
    """Aggregate scores across all models and verticals."""
    model_scores = []

    for model in models:
        vertical_scores = {}
        all_task_scores = []
        total_latency = 0
        task_count = 0
        hurdle_pass_count = 0

        for vertical in verticals:
            task_results = load_task_results(results_dir, model, vertical, run)

            if not task_results:
                vertical_scores[vertical] = 0.0
                continue

            vertical_task_scores = []
            for result in task_results:
                score_data = calculate_task_score(result["grounded_response"])
                vertical_task_scores.append(score_data["total_score"])
                all_task_scores.append(score_data)
                total_latency += score_data.get("latency_ms", 0)
                task_count += 1
                if score_data["hurdle_passed"]:
                    hurdle_pass_count += 1

            vertical_scores[vertical] = (
                sum(vertical_task_scores) / len(vertical_task_scores) * 100
                if vertical_task_scores else 0.0
            )

        # Calculate overall VCI score (average across verticals)
        vci_score = (
            sum(vertical_scores.values()) / len(vertical_scores)
            if vertical_scores else 0.0
        )

        # Calculate component averages
        grounded_scores = [s["component_scores"].get("grounded", 0) for s in all_task_scores]
        helpfulness_scores = [s["component_scores"].get("helpfulness", 0) for s in all_task_scores]

        model_scores.append(ModelScore(
            model=model,
            vci_score=vci_score,
            vertical_scores=vertical_scores,
            hurdle_pass_rate=hurdle_pass_count / task_count * 100 if task_count > 0 else 0,
            grounded_score=sum(grounded_scores) / len(grounded_scores) * 100 if grounded_scores else 0,
            helpfulness_score=sum(helpfulness_scores) / len(helpfulness_scores) * 100 if helpfulness_scores else 0,
            tasks_completed=task_count,
            tasks_failed=0,  # TODO: Track failures
            avg_latency_ms=total_latency / task_count if task_count > 0 else 0
        ))

    return model_scores


def generate_leaderboard(model_scores: List[ModelScore]) -> List[LeaderboardEntry]:
    """Generate ranked leaderboard from model scores."""
    sorted_scores = sorted(model_scores, key=lambda x: x.vci_score, reverse=True)

    leaderboard = []
    for rank, score in enumerate(sorted_scores, 1):
        leaderboard.append(LeaderboardEntry(
            rank=rank,
            model=score.model,
            vci_score=round(score.vci_score, 1),
            vertical_scores={k: round(v, 1) for k, v in score.vertical_scores.items()},
            hurdle_pass_rate=round(score.hurdle_pass_rate, 1)
        ))

    return leaderboard


def format_leaderboard_text(leaderboard: List[LeaderboardEntry], verticals: List[str]) -> str:
    """Format leaderboard as text table."""
    lines = []

    # Header
    header = f"{'Rank':<6} {'Model':<20} {'VCI Score':<12}"
    for v in verticals:
        header += f" {v.capitalize():<10}"
    header += f" {'Hurdle %':<10}"
    lines.append(header)
    lines.append("=" * len(header))

    for entry in leaderboard:
        row = f"{entry.rank:<6} {entry.model:<20} {entry.vci_score:<12.1f}"
        for v in verticals:
            row += f" {entry.vertical_scores.get(v, 0):<10.1f}"
        row += f" {entry.hurdle_pass_rate:<10.1f}"
        lines.append(row)

    return "\n".join(lines)


def format_leaderboard_markdown(leaderboard: List[LeaderboardEntry], verticals: List[str]) -> str:
    """Format leaderboard as Markdown table."""
    lines = []

    # Header
    header = "| Rank | Model | VCI Score |"
    for v in verticals:
        header += f" {v.capitalize()} |"
    header += " Hurdle % |"
    lines.append(header)

    # Separator
    sep = "|------|-------|-----------|"
    for _ in verticals:
        sep += "----------|"
    sep += "----------|"
    lines.append(sep)

    for entry in leaderboard:
        row = f"| {entry.rank} | {entry.model} | {entry.vci_score:.1f} |"
        for v in verticals:
            row += f" {entry.vertical_scores.get(v, 0):.1f} |"
        row += f" {entry.hurdle_pass_rate:.1f} |"
        lines.append(row)

    return "\n".join(lines)


def save_results(
    model_scores: List[ModelScore],
    leaderboard: List[LeaderboardEntry],
    output_dir: Path,
    run: int
):
    """Save aggregated results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model scores as JSON
    scores_file = output_dir / f"model_scores_run_{run}.json"
    with open(scores_file, "w") as f:
        json.dump([asdict(s) for s in model_scores], f, indent=2)

    # Save leaderboard as JSON
    leaderboard_file = output_dir / f"leaderboard_run_{run}.json"
    with open(leaderboard_file, "w") as f:
        json.dump([asdict(e) for e in leaderboard], f, indent=2)

    # Save leaderboard as Markdown
    markdown_file = output_dir / f"leaderboard_run_{run}.md"
    with open(markdown_file, "w") as f:
        f.write("# VCI Benchmark Leaderboard\n\n")
        f.write(f"Run: {run}\n\n")
        f.write(format_leaderboard_markdown(leaderboard, VERTICALS))

    return scores_file, leaderboard_file, markdown_file


def main():
    parser = argparse.ArgumentParser(description="Aggregate VCI benchmark results")
    parser.add_argument("--run", type=int, default=1, help="Run number to aggregate")
    parser.add_argument("--format", choices=["text", "markdown", "json"], default="text",
                       help="Output format")
    parser.add_argument("--output-dir", type=Path, default=Path(SETTINGS.results_dir),
                       help="Output directory for results")

    args = parser.parse_args()

    results_dir = Path(SETTINGS.results_dir)

    # Discover models from results directory
    models = []
    if results_dir.exists():
        for model_dir in results_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("."):
                models.append(model_dir.name)

    if not models:
        print("No results found. Run the benchmark first.")
        return

    print(f"Found results for {len(models)} models")

    # Aggregate scores
    model_scores = aggregate_model_scores(results_dir, models, VERTICALS, args.run)

    # Generate leaderboard
    leaderboard = generate_leaderboard(model_scores)

    # Output
    if args.format == "text":
        print("\n" + format_leaderboard_text(leaderboard, VERTICALS))
    elif args.format == "markdown":
        print("\n" + format_leaderboard_markdown(leaderboard, VERTICALS))
    elif args.format == "json":
        print(json.dumps([asdict(e) for e in leaderboard], indent=2))

    # Save results
    scores_file, leaderboard_file, markdown_file = save_results(
        model_scores, leaderboard, args.output_dir, args.run
    )

    print(f"\nResults saved to:")
    print(f"  - {scores_file}")
    print(f"  - {leaderboard_file}")
    print(f"  - {markdown_file}")


if __name__ == "__main__":
    main()
