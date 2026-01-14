#!/usr/bin/env python3
"""Run VCI benchmark with parallel model execution."""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from vci.configs.settings import VERTICALS
from vci.pipeline.runner import run_all_tasks

# All models to benchmark
KEY_MODELS = [
    # OpenAI
    "gpt-4o", "gpt-4o-mini",
    "gpt-5", "gpt-5.1", "gpt-5.2",
    "o3", "o3-mini", "o3-pro",
    # Anthropic
    "claude-3.5-sonnet",
    "sonnet-4", "sonnet-4.5",
    "opus-4", "opus-4.1", "opus-4.5",
    # Google
    "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro",
    "gemini-3-flash", "gemini-3-pro",
    # Perplexity
    "sonar", "sonar-pro",
    # Others
    "deepseek-chat", "grok-3",
]

RUN_NUMBER = 5

def run_model_vertical(model, vertical):
    """Run one model on one vertical."""
    try:
        stats = run_all_tasks(vertical, model, RUN_NUMBER, dataset="dev")
        return {"model": model, "vertical": vertical, "success": True, "tasks": stats.get('completed', 0)}
    except Exception as e:
        return {"model": model, "vertical": vertical, "success": False, "error": str(e)}

def main():
    print(f"\n{'='*70}")
    print(f"VCI PARALLEL BENCHMARK RUN {RUN_NUMBER}")
    print(f"Models: {len(KEY_MODELS)}")
    print(f"Verticals: {VERTICALS}")
    print(f"{'='*70}\n")

    # Create all model-vertical combinations
    jobs = [(m, v) for m in KEY_MODELS for v in VERTICALS]

    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:  # 3 parallel
        futures = {executor.submit(run_model_vertical, m, v): (m, v) for m, v in jobs}

        for future in as_completed(futures):
            m, v = futures[future]
            try:
                result = future.result()
                status = "OK" if result["success"] else "FAIL"
                tasks = result.get("tasks", 0)
                print(f"[{status}] {m}/{v}: {tasks} tasks", flush=True)
                results.append(result)
            except Exception as e:
                print(f"[ERR] {m}/{v}: {e}", flush=True)

    # Summary
    successful = sum(1 for r in results if r.get("success"))
    total_tasks = sum(r.get("tasks", 0) for r in results)

    print(f"\n{'='*70}")
    print(f"COMPLETE: {successful}/{len(jobs)} model-verticals")
    print(f"Total tasks: {total_tasks}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
