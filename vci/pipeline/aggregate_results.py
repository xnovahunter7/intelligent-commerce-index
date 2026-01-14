#!/usr/bin/env python3
"""
VCI Results Aggregation

Aggregates benchmark results across models and verticals to produce:
- VCI scores per model
- Leaderboard rankings
- Detailed breakdowns by vertical and criteria type
- Historical run tracking and comparison

Usage:
    python -m vci.pipeline.aggregate_results --run 1
    python -m vci.pipeline.aggregate_results --run 1 --format markdown
    python -m vci.pipeline.aggregate_results --history  # Show all historical runs
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
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


@dataclass
class RunMetadata:
    """Metadata for a benchmark run."""
    run_id: int
    timestamp: str
    models_count: int
    tasks_count: int
    verticals: List[str]
    dataset: str = "dev"


@dataclass
class HistoricalRun:
    """A single historical run with its results."""
    metadata: RunMetadata
    leaderboard: List[LeaderboardEntry]
    top_model: str
    top_score: float


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

        # Load autograder results (Stage 3) - this has the actual scores
        autograder_file = task_dir / "3_autograder_results.json"
        grounded_file = task_dir / "1_grounded_response.json"

        if autograder_file.exists():
            with open(autograder_file) as f:
                autograder_result = json.load(f)

            # Get latency from grounded response
            latency_ms = 0
            if grounded_file.exists():
                with open(grounded_file) as f:
                    grounded = json.load(f)
                    latency_ms = grounded.get("latency_ms", 0)

            results.append({
                "task_id": task_dir.name.replace("task_", ""),
                "model": model,
                "vertical": vertical,
                "autograder_result": autograder_result,
                "latency_ms": latency_ms,
            })
        elif grounded_file.exists():
            # Fallback: only Stage 1 complete, use simplified scoring
            with open(grounded_file) as f:
                grounded_response = json.load(f)
                results.append({
                    "task_id": task_dir.name.replace("task_", ""),
                    "model": model,
                    "vertical": vertical,
                    "grounded_response": grounded_response,
                    "latency_ms": grounded_response.get("latency_ms", 0),
                })

    return results


def calculate_task_score(task_result: Dict) -> Dict[str, Any]:
    """
    Calculate score for a single task.

    Uses autograder results if available, falls back to simplified scoring.
    """
    # If we have autograder results, use them directly
    if "autograder_result" in task_result:
        ag = task_result["autograder_result"]
        return {
            "hurdle_passed": ag.get("hurdle_passed", False),
            "total_score": ag.get("total_score", 0.0),
            "component_scores": ag.get("component_scores", {}),
            "latency_ms": task_result.get("latency_ms", 0),
            "criterion_results": ag.get("criterion_results", [])
        }

    # Fallback: simplified scoring from grounded_response (legacy)
    grounded_response = task_result.get("grounded_response", {})
    criteria = grounded_response.get("criteria", [])

    if not criteria:
        return {
            "hurdle_passed": True,  # Assume pass if no criteria
            "total_score": 0.5,
            "component_scores": {"grounded": 0.5, "helpfulness": 0.5},
            "latency_ms": task_result.get("latency_ms", 0)
        }

    # Simple scoring: assume all pass for legacy results
    return {
        "hurdle_passed": True,
        "total_score": 0.5,
        "component_scores": {"grounded": 0.5, "helpfulness": 0.5},
        "latency_ms": task_result.get("latency_ms", 0)
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
                score_data = calculate_task_score(result)  # Pass full result, not grounded_response
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
        safety_scores = [s["component_scores"].get("safety", 0) for s in all_task_scores]
        completeness_scores = [s["component_scores"].get("completeness", 0) for s in all_task_scores]

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
    header = f"{'Rank':<6} {'Model':<25} {'VCI Score':<12}"
    for v in verticals:
        header += f" {v.capitalize():<10}"
    header += f" {'Hurdle %':<10}"
    lines.append(header)
    lines.append("=" * len(header))

    for entry in leaderboard:
        row = f"{entry.rank:<6} {entry.model:<25} {entry.vci_score:<12.1f}"
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


def discover_runs(results_dir: Path) -> List[int]:
    """Discover all run numbers from results directory."""
    runs = set()
    if not results_dir.exists():
        return []

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        for vertical_dir in model_dir.iterdir():
            if not vertical_dir.is_dir():
                continue
            for run_dir in vertical_dir.iterdir():
                if run_dir.is_dir() and run_dir.name.startswith("run_"):
                    try:
                        run_num = int(run_dir.name.replace("run_", ""))
                        runs.add(run_num)
                    except ValueError:
                        continue

    return sorted(runs)


def load_run_history(results_dir: Path) -> List[HistoricalRun]:
    """Load all historical runs from the history file."""
    history_file = results_dir / "run_history.json"
    if not history_file.exists():
        return []

    with open(history_file) as f:
        data = json.load(f)

    runs = []
    for run_data in data.get("runs", []):
        metadata = RunMetadata(**run_data["metadata"])
        leaderboard = [LeaderboardEntry(**e) for e in run_data["leaderboard"]]
        runs.append(HistoricalRun(
            metadata=metadata,
            leaderboard=leaderboard,
            top_model=run_data["top_model"],
            top_score=run_data["top_score"]
        ))

    return runs


def save_run_history(results_dir: Path, runs: List[HistoricalRun]):
    """Save run history to file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    history_file = results_dir / "run_history.json"

    data = {
        "last_updated": datetime.now().isoformat(),
        "total_runs": len(runs),
        "runs": [
            {
                "metadata": asdict(run.metadata),
                "leaderboard": [asdict(e) for e in run.leaderboard],
                "top_model": run.top_model,
                "top_score": run.top_score
            }
            for run in runs
        ]
    }

    with open(history_file, "w") as f:
        json.dump(data, f, indent=2)


def add_to_history(
    results_dir: Path,
    run: int,
    leaderboard: List[LeaderboardEntry],
    model_scores: List[ModelScore]
):
    """Add current run to history."""
    history = load_run_history(results_dir)

    # Check if run already exists
    existing_run_ids = {r.metadata.run_id for r in history}
    if run in existing_run_ids:
        # Update existing run
        history = [r for r in history if r.metadata.run_id != run]

    # Calculate totals
    total_tasks = sum(s.tasks_completed for s in model_scores)

    # Create metadata
    metadata = RunMetadata(
        run_id=run,
        timestamp=datetime.now().isoformat(),
        models_count=len(model_scores),
        tasks_count=total_tasks,
        verticals=VERTICALS,
        dataset="dev"
    )

    # Get top model
    top_model = leaderboard[0].model if leaderboard else "N/A"
    top_score = leaderboard[0].vci_score if leaderboard else 0.0

    # Add to history
    history.append(HistoricalRun(
        metadata=metadata,
        leaderboard=leaderboard,
        top_model=top_model,
        top_score=top_score
    ))

    # Sort by run_id
    history.sort(key=lambda r: r.metadata.run_id)

    # Save
    save_run_history(results_dir, history)


def format_history_text(history: List[HistoricalRun]) -> str:
    """Format run history as text table."""
    if not history:
        return "No historical runs found."

    lines = []
    lines.append("=" * 90)
    lines.append("VCI BENCHMARK RUN HISTORY")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"{'Run':<6} {'Timestamp':<22} {'Models':<8} {'Tasks':<8} {'Top Model':<25} {'Score':<8}")
    lines.append("-" * 90)

    for run in history:
        timestamp = run.metadata.timestamp[:19].replace("T", " ")
        lines.append(
            f"{run.metadata.run_id:<6} "
            f"{timestamp:<22} "
            f"{run.metadata.models_count:<8} "
            f"{run.metadata.tasks_count:<8} "
            f"{run.top_model:<25} "
            f"{run.top_score:<8.1f}"
        )

    lines.append("-" * 90)
    lines.append(f"Total runs: {len(history)}")

    return "\n".join(lines)


def format_history_markdown(history: List[HistoricalRun]) -> str:
    """Format run history as Markdown."""
    if not history:
        return "No historical runs found."

    lines = []
    lines.append("# VCI Benchmark Run History")
    lines.append("")
    lines.append("| Run | Timestamp | Models | Tasks | Top Model | Score |")
    lines.append("|-----|-----------|--------|-------|-----------|-------|")

    for run in history:
        timestamp = run.metadata.timestamp[:19].replace("T", " ")
        lines.append(
            f"| {run.metadata.run_id} | {timestamp} | "
            f"{run.metadata.models_count} | {run.metadata.tasks_count} | "
            f"{run.top_model} | {run.top_score:.1f} |"
        )

    lines.append("")
    lines.append(f"**Total runs:** {len(history)}")

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
        f.write(f"**Run:** {run}\n")
        f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Models:** {len(model_scores)}\n\n")
        f.write(format_leaderboard_markdown(leaderboard, VERTICALS))

    # Add to history
    add_to_history(output_dir, run, leaderboard, model_scores)

    # Generate combined history markdown
    history = load_run_history(output_dir)
    history_file = output_dir / "HISTORY.md"
    with open(history_file, "w") as f:
        f.write(format_history_markdown(history))
        f.write("\n\n---\n\n")
        f.write("## Latest Run Details\n\n")
        f.write(f"### Run {run}\n\n")
        f.write(format_leaderboard_markdown(leaderboard, VERTICALS))

    return scores_file, leaderboard_file, markdown_file


def main():
    parser = argparse.ArgumentParser(description="Aggregate VCI benchmark results")
    parser.add_argument("--run", type=int, default=1, help="Run number to aggregate")
    parser.add_argument("--format", choices=["text", "markdown", "json"], default="text",
                       help="Output format")
    parser.add_argument("--output-dir", type=Path, default=Path(SETTINGS.results_dir),
                       help="Output directory for results")
    parser.add_argument("--history", action="store_true",
                       help="Show all historical runs")
    parser.add_argument("--list-runs", action="store_true",
                       help="List all available run numbers")

    args = parser.parse_args()

    results_dir = Path(SETTINGS.results_dir)

    # Handle --list-runs
    if args.list_runs:
        runs = discover_runs(results_dir)
        if runs:
            print(f"Available runs: {', '.join(map(str, runs))}")
        else:
            print("No runs found.")
        return

    # Handle --history
    if args.history:
        history = load_run_history(results_dir)
        if args.format == "markdown":
            print(format_history_markdown(history))
        else:
            print(format_history_text(history))
        return

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
    print(f"  - {args.output_dir / 'run_history.json'}")
    print(f"  - {args.output_dir / 'HISTORY.md'}")


if __name__ == "__main__":
    main()
