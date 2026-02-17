#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def run_command(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def summarize_metrics(metrics_file: Path) -> dict[str, str]:
    with metrics_file.open("r", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {
            "step_count": "0",
            "avg_step_time_sec": "0.0",
            "avg_estimated_tokens_per_sec": "0.0",
            "peak_cuda_memory_allocated_gb": "0.0",
        }

    step_count = len(rows)
    avg_step_time = sum(float(row["step_time_sec"]) for row in rows) / step_count
    avg_tps = sum(float(row["estimated_tokens_per_sec"]) for row in rows) / step_count
    peak_mem = max(float(row["cuda_peak_memory_allocated_gb"]) for row in rows)
    return {
        "step_count": str(step_count),
        "avg_step_time_sec": f"{avg_step_time:.4f}",
        "avg_estimated_tokens_per_sec": f"{avg_tps:.2f}",
        "peak_cuda_memory_allocated_gb": f"{peak_mem:.4f}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL feasibility matrix on one GPU.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--frameworks", type=str, default="unsloth,trl")
    parser.add_argument("--modes", type=str, default="grpo,gspo")
    parser.add_argument("--num-generations", type=str, default="2,4,8")
    parser.add_argument("--batch-sizes", type=str, default="1,2")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--dataset-size", type=int, default=4000)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/matrix"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_file = args.output_root / "summary.csv"
    frameworks = [item.strip() for item in args.frameworks.split(",") if item.strip()]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    num_generations_values = parse_int_list(args.num_generations)
    batch_sizes = parse_int_list(args.batch_sizes)

    rows_to_write: list[dict[str, str]] = []
    for framework in frameworks:
        script = Path("scripts/run_unsloth_rl.py" if framework == "unsloth" else "scripts/run_trl_rl.py")
        for mode in modes:
            for num_generations in num_generations_values:
                for batch_size in batch_sizes:
                    run_name = f"{framework}_{mode}_k{num_generations}_b{batch_size}"
                    output_dir = args.output_root / run_name
                    metrics_file = output_dir / "metrics.csv"

                    cmd = [
                        sys.executable,
                        str(script),
                        "--model-name",
                        args.model_name,
                        "--mode",
                        mode,
                        "--output-dir",
                        str(output_dir),
                        "--metrics-file",
                        str(metrics_file),
                        "--dataset-size",
                        str(args.dataset_size),
                        "--max-steps",
                        str(args.max_steps),
                        "--batch-size",
                        str(batch_size),
                        "--grad-accum",
                        str(args.grad_accum),
                        "--num-generations",
                        str(num_generations),
                        "--max-prompt-length",
                        str(args.max_prompt_length),
                        "--max-completion-length",
                        str(args.max_completion_length),
                    ]

                    run_command(cmd)
                    summary = summarize_metrics(metrics_file)
                    rows_to_write.append(
                        {
                            "run_name": run_name,
                            "framework": framework,
                            "mode": mode,
                            "num_generations": str(num_generations),
                            "batch_size": str(batch_size),
                            **summary,
                        }
                    )

    with summary_file.open("w", newline="") as file:
        fieldnames = [
            "run_name",
            "framework",
            "mode",
            "num_generations",
            "batch_size",
            "step_count",
            "avg_step_time_sec",
            "avg_estimated_tokens_per_sec",
            "peak_cuda_memory_allocated_gb",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)
    print(f"Summary written to: {summary_file}")


if __name__ == "__main__":
    main()
