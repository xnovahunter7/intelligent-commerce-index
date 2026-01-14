#!/usr/bin/env python3
"""
VCI Human Baseline Comparison

Creates and manages human baseline annotations for comparing model performance.
Allows experts to provide ground truth for selected tasks.

Usage:
    # Create baseline template
    python -m vci.pipeline.human_baseline --create --task VCI-ELEC-001

    # Score a model against human baseline
    python -m vci.pipeline.human_baseline --score --task VCI-ELEC-001 --model gpt-4o

    # Generate report for all baseline tasks
    python -m vci.pipeline.human_baseline --report
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from ..configs.settings import SETTINGS, VERTICALS


BASELINE_DIR = Path(SETTINGS.results_dir) / "human_baseline"


@dataclass
class HumanCriterionScore:
    """Human-annotated score for a criterion."""
    criterion_id: str
    human_score: float  # 0.0, 0.5, or 1.0
    reasoning: str
    confidence: str  # "high", "medium", "low"


@dataclass
class HumanBaseline:
    """Human baseline annotation for a task."""
    task_id: str
    vertical: str
    annotator: str
    timestamp: str
    prompt: str
    ideal_response_summary: str  # What an ideal response would include
    criterion_scores: List[HumanCriterionScore]
    notes: str

    def save(self, path: Path):
        with open(path, "w") as f:
            data = {
                "task_id": self.task_id,
                "vertical": self.vertical,
                "annotator": self.annotator,
                "timestamp": self.timestamp,
                "prompt": self.prompt,
                "ideal_response_summary": self.ideal_response_summary,
                "criterion_scores": [asdict(c) for c in self.criterion_scores],
                "notes": self.notes
            }
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "HumanBaseline":
        with open(path) as f:
            data = json.load(f)
        return cls(
            task_id=data["task_id"],
            vertical=data["vertical"],
            annotator=data["annotator"],
            timestamp=data["timestamp"],
            prompt=data["prompt"],
            ideal_response_summary=data["ideal_response_summary"],
            criterion_scores=[HumanCriterionScore(**c) for c in data["criterion_scores"]],
            notes=data["notes"]
        )


def create_baseline_template(task_id: str, vertical: str = None) -> Path:
    """Create a template for human annotation."""
    from .runner import load_tasks_for_vertical

    # Find the task
    task = None
    for v in VERTICALS:
        if vertical and v != vertical:
            continue
        tasks = load_tasks_for_vertical(v)
        for t in tasks:
            if t.task_id == task_id:
                task = t
                vertical = v
                break
        if task:
            break

    if not task:
        raise ValueError(f"Task {task_id} not found")

    # Create baseline directory
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    # Generate template
    criterion_scores = []
    for crit in task.criteria:
        criterion_scores.append(HumanCriterionScore(
            criterion_id=crit["criterion_id"],
            human_score=0.0,  # To be filled by human
            reasoning="[TODO: Explain why this score]",
            confidence="medium"
        ))

    baseline = HumanBaseline(
        task_id=task_id,
        vertical=vertical,
        annotator="[YOUR NAME]",
        timestamp=datetime.now().isoformat(),
        prompt=task.prompt,
        ideal_response_summary="[TODO: Describe what an ideal response would include]",
        criterion_scores=criterion_scores,
        notes="[TODO: Any additional notes about grading this task]"
    )

    output_path = BASELINE_DIR / f"{task_id}_baseline.json"
    baseline.save(output_path)

    print(f"Created baseline template: {output_path}")
    print("\nPlease edit the file to fill in:")
    print("  - annotator: Your name")
    print("  - ideal_response_summary: What an ideal response should include")
    print("  - For each criterion:")
    print("    - human_score: 0.0 (fail), 0.5 (partial), or 1.0 (pass)")
    print("    - reasoning: Why you gave this score")
    print("    - confidence: high/medium/low")
    print("  - notes: Any grading notes")

    return output_path


def load_model_result(task_id: str, model: str, run: int = 1) -> Optional[Dict]:
    """Load autograder result for a model."""
    results_dir = Path(SETTINGS.results_dir)
    model_dir = model.replace("/", "_")

    for vertical in VERTICALS:
        ag_file = results_dir / model_dir / vertical / f"run_{run}" / f"task_{task_id}" / "3_autograder_results.json"
        if ag_file.exists():
            with open(ag_file) as f:
                return json.load(f)

    return None


def compare_to_baseline(
    task_id: str,
    model: str,
    run: int = 1
) -> Optional[Dict[str, Any]]:
    """Compare model result to human baseline."""
    baseline_path = BASELINE_DIR / f"{task_id}_baseline.json"

    if not baseline_path.exists():
        print(f"No baseline found for {task_id}")
        return None

    baseline = HumanBaseline.load(baseline_path)
    model_result = load_model_result(task_id, model, run)

    if not model_result:
        print(f"No model result found for {task_id} / {model} / run_{run}")
        return None

    # Build comparison
    comparison = {
        "task_id": task_id,
        "model": model,
        "run": run,
        "human_annotator": baseline.annotator,
        "criteria_comparison": [],
        "summary": {}
    }

    human_scores = {c.criterion_id: c for c in baseline.criterion_scores}
    model_scores = {c["criterion_id"]: c for c in model_result.get("criterion_results", [])}

    agreements = 0
    disagreements = 0
    human_total = 0
    model_total = 0

    for crit_id in human_scores:
        human = human_scores[crit_id]
        model = model_scores.get(crit_id, {})

        human_score = human.human_score
        model_score = model.get("score", 0.0)

        # Normalize model score (could be -1.0)
        model_score_normalized = max(0, model_score)

        # Check agreement (within 0.25 tolerance for partial credit)
        agree = abs(human_score - model_score_normalized) <= 0.25

        if agree:
            agreements += 1
        else:
            disagreements += 1

        human_total += human_score
        model_total += model_score_normalized

        comparison["criteria_comparison"].append({
            "criterion_id": crit_id,
            "human_score": human_score,
            "human_reasoning": human.reasoning,
            "human_confidence": human.confidence,
            "model_score": model_score,
            "model_reasoning": model.get("reasoning", "N/A"),
            "agrees": agree
        })

    total_criteria = len(human_scores)
    agreement_rate = (agreements / total_criteria * 100) if total_criteria > 0 else 0

    comparison["summary"] = {
        "total_criteria": total_criteria,
        "agreements": agreements,
        "disagreements": disagreements,
        "agreement_rate": round(agreement_rate, 1),
        "human_total_score": round(human_total / total_criteria, 3) if total_criteria > 0 else 0,
        "model_total_score": round(model_total / total_criteria, 3) if total_criteria > 0 else 0
    }

    return comparison


def generate_report() -> Dict[str, Any]:
    """Generate report comparing all baseline tasks across models."""
    if not BASELINE_DIR.exists():
        print("No baselines found.")
        return {}

    baseline_files = list(BASELINE_DIR.glob("*_baseline.json"))
    if not baseline_files:
        print("No baseline files found.")
        return {}

    # Find all models with results
    results_dir = Path(SETTINGS.results_dir)
    models = []
    for d in results_dir.iterdir():
        if d.is_dir() and not d.name.startswith(".") and d.name != "human_baseline":
            models.append(d.name)

    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_tasks": len(baseline_files),
        "models_tested": len(models),
        "model_results": {},
        "task_details": {}
    }

    for model in models:
        model_agreements = 0
        model_total = 0

        for baseline_file in baseline_files:
            task_id = baseline_file.stem.replace("_baseline", "")
            comparison = compare_to_baseline(task_id, model)

            if comparison:
                model_agreements += comparison["summary"]["agreements"]
                model_total += comparison["summary"]["total_criteria"]

                if task_id not in report["task_details"]:
                    report["task_details"][task_id] = {}
                report["task_details"][task_id][model] = comparison["summary"]

        if model_total > 0:
            report["model_results"][model] = {
                "agreement_rate": round(model_agreements / model_total * 100, 1),
                "criteria_tested": model_total
            }

    return report


def print_report(report: Dict[str, Any]):
    """Print baseline comparison report."""
    if not report:
        return

    print("\n" + "=" * 70)
    print("HUMAN BASELINE COMPARISON REPORT")
    print("=" * 70)

    print(f"\nBaseline tasks: {report.get('baseline_tasks', 0)}")
    print(f"Models tested: {report.get('models_tested', 0)}")

    if report.get("model_results"):
        print("\n" + "-" * 70)
        print(f"{'Model':<30} {'Agreement Rate':<20} {'Criteria':<15}")
        print("-" * 70)

        for model, data in sorted(report["model_results"].items(), key=lambda x: -x[1]["agreement_rate"]):
            print(f"{model:<30} {data['agreement_rate']:<20.1f}% {data['criteria_tested']:<15}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Manage human baseline comparisons")
    parser.add_argument("--create", action="store_true", help="Create baseline template")
    parser.add_argument("--score", action="store_true", help="Score model against baseline")
    parser.add_argument("--report", action="store_true", help="Generate comparison report")
    parser.add_argument("--task", help="Task ID")
    parser.add_argument("--model", default="gpt-4o", help="Model to compare")
    parser.add_argument("--run", type=int, default=1, help="Run number")
    parser.add_argument("--output", type=Path, help="Output file for results")

    args = parser.parse_args()

    if args.create:
        if not args.task:
            print("--task required for --create")
            return
        create_baseline_template(args.task)

    elif args.score:
        if not args.task:
            print("--task required for --score")
            return
        comparison = compare_to_baseline(args.task, args.model, args.run)
        if comparison:
            print(json.dumps(comparison, indent=2))
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(comparison, f, indent=2)

    elif args.report:
        report = generate_report()
        print_report(report)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
