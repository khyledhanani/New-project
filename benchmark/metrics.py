from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


@dataclass(frozen=True)
class ThroughputEstimate:
    prompt_tokens: int
    completion_tokens: int
    num_generations: int
    train_batch_size: int

    @property
    def estimated_tokens_per_step(self) -> int:
        return (
            (self.prompt_tokens + self.completion_tokens)
            * self.num_generations
            * self.train_batch_size
        )


class CSVStepMetricsCallback(TrainerCallback):
    def __init__(self, output_file: Path, estimate: ThroughputEstimate) -> None:
        self.output_file = output_file
        self.estimate = estimate
        self._step_start_time: float | None = None
        self._initialized = False

    def _init_csv(self) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "step",
                    "step_time_sec",
                    "estimated_tokens_per_step",
                    "estimated_tokens_per_sec",
                    "cuda_memory_allocated_gb",
                    "cuda_memory_reserved_gb",
                    "cuda_peak_memory_allocated_gb",
                ]
            )
        self._initialized = True

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        del args, state, control, kwargs
        if not self._initialized:
            self._init_csv()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        del args, state, control, kwargs
        self._step_start_time = time.perf_counter()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        del args, control, kwargs
        if self._step_start_time is None:
            return
        step_time = time.perf_counter() - self._step_start_time
        estimated_tokens = self.estimate.estimated_tokens_per_step
        estimated_tps = estimated_tokens / step_time if step_time > 0 else 0.0

        allocated = 0.0
        reserved = 0.0
        peak_allocated = 0.0
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)

        with self.output_file.open("a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    state.global_step,
                    f"{step_time:.6f}",
                    estimated_tokens,
                    f"{estimated_tps:.2f}",
                    f"{allocated:.4f}",
                    f"{reserved:.4f}",
                    f"{peak_allocated:.4f}",
                ]
            )
