#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def parse_run_name(run_name: str) -> tuple[str, str, str, str]:
    # Expected format: framework_mode_kX_bY[_seedZ]
    parts = run_name.split("_")
    framework = parts[0] if len(parts) > 0 else "unknown"
    mode = parts[1] if len(parts) > 1 else "unknown"
    k = next((p for p in parts if p.startswith("k")), "k?")
    b = next((p for p in parts if p.startswith("b")), "b?")
    return framework, mode, k, b


def read_rows(summary_file: Path) -> list[dict[str, str]]:
    with summary_file.open("r", newline="") as file:
        return list(csv.DictReader(file))


def to_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0"))
    except ValueError:
        return 0.0


def group_key(row: dict[str, str]) -> str:
    run_name = row.get("run_name", "")
    framework, mode, k, b = parse_run_name(run_name)
    if framework != "unknown" and mode != "unknown" and k != "k?" and b != "b?":
        return f"{framework}_{mode}_{k}_{b}"
    framework = row.get("framework", framework)
    mode = row.get("mode", mode)
    k = f"k{row.get('num_generations', '?')}"
    b = f"b{row.get('batch_size', '?')}"
    return f"{framework}_{mode}_{k}_{b}"


def build_aggregate(rows: list[dict[str, str]], vram_cap: float) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    aggregated: list[dict[str, str]] = []
    for key, group in grouped.items():
        tps_values = sorted(to_float(row, "avg_estimated_tokens_per_sec") for row in group)
        step_values = sorted(to_float(row, "avg_step_time_sec") for row in group)
        peak_values = sorted(to_float(row, "peak_cuda_memory_allocated_gb") for row in group)
        repeats = len(group)

        median_tps = statistics.median(tps_values) if tps_values else 0.0
        p10_tps = percentile(tps_values, 0.10)
        p90_tps = percentile(tps_values, 0.90)
        median_step = statistics.median(step_values) if step_values else 0.0
        max_peak = max(peak_values) if peak_values else 0.0
        stable_under_cap = max_peak <= vram_cap

        framework, mode, k, b = parse_run_name(key)
        aggregated.append(
            {
                "config": key,
                "framework": framework,
                "mode": mode,
                "num_generations": k[1:],
                "batch_size": b[1:],
                "repeats": str(repeats),
                "median_tps": f"{median_tps:.2f}",
                "p10_tps": f"{p10_tps:.2f}",
                "p90_tps": f"{p90_tps:.2f}",
                "median_step_time_sec": f"{median_step:.4f}",
                "max_peak_vram_gb": f"{max_peak:.4f}",
                "within_vram_cap": "yes" if stable_under_cap else "no",
            }
        )
    aggregated.sort(
        key=lambda row: (
            row["within_vram_cap"] != "yes",
            -float(row["median_tps"]),
            float(row["median_step_time_sec"]),
        )
    )
    return aggregated


def write_csv(rows: list[dict[str, str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_file.open("w", newline="") as file:
            file.write("")
        return
    with output_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], output_file: Path, top_n: int) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    selected = rows[:top_n]
    headers = [
        "config",
        "framework",
        "mode",
        "num_generations",
        "batch_size",
        "repeats",
        "median_tps",
        "p10_tps",
        "p90_tps",
        "max_peak_vram_gb",
        "within_vram_cap",
    ]
    with output_file.open("w") as file:
        file.write("# Ranked RL Feasibility Results\n\n")
        file.write("| " + " | ".join(headers) + " |\n")
        file.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in selected:
            file.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def maybe_plot(rows: list[dict[str, str]], output_dir: Path) -> str:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return "matplotlib not installed; skipping plots."

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered = [r for r in rows if r["within_vram_cap"] == "yes"]
    if not filtered:
        filtered = rows
    labels = [r["config"] for r in filtered]
    tps = [float(r["median_tps"]) for r in filtered]
    vram = [float(r["max_peak_vram_gb"]) for r in filtered]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(labels)), tps)
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
    plt.ylabel("Median estimated tokens/sec")
    plt.tight_layout()
    tps_path = output_dir / "ranked_median_tps.png"
    plt.savefig(tps_path, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(labels)), vram, color="orange")
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
    plt.ylabel("Max peak VRAM (GB)")
    plt.tight_layout()
    vram_path = output_dir / "ranked_peak_vram.png"
    plt.savefig(vram_path, dpi=150)
    plt.close()
    return f"wrote {tps_path} and {vram_path}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RL benchmark summary.csv.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/matrix/summary.csv"),
        help="Path to matrix summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/matrix/analysis"),
        help="Directory for ranked outputs.",
    )
    parser.add_argument(
        "--vram-cap-gb",
        type=float,
        default=15.5,
        help="Configs above this peak VRAM are marked not feasible.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Rows to include in markdown table.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate PNG charts if matplotlib is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    aggregated = build_aggregate(rows, args.vram_cap_gb)
    ranked_csv = args.output_dir / "ranked_configs.csv"
    ranked_md = args.output_dir / "ranked_configs.md"

    write_csv(aggregated, ranked_csv)
    write_markdown(aggregated, ranked_md, args.top_n)

    print(f"Input rows: {len(rows)}")
    print(f"Aggregated configs: {len(aggregated)}")
    print(f"Wrote: {ranked_csv}")
    print(f"Wrote: {ranked_md}")
    if args.plot:
        print(maybe_plot(aggregated, args.output_dir))


if __name__ == "__main__":
    main()
