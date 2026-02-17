#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import torch
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL

from benchmark.data import make_arithmetic_samples, to_hf_dataset
from benchmark.metrics import CSVStepMetricsCallback, ThroughputEstimate
from benchmark.reward import exact_arithmetic_reward


def filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    allowed = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in allowed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsloth GRPO/GSPO QLoRA benchmark.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mode", choices=["grpo", "gspo"], default="grpo")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/unsloth"))
    parser.add_argument("--metrics-file", type=Path, default=Path("outputs/unsloth/metrics.csv"))
    parser.add_argument("--dataset-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    PatchFastRL("GRPO", FastLanguageModel)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=dtype,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    train_dataset = to_hf_dataset(make_arithmetic_samples(size=args.dataset_size, seed=args.seed))
    config_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_strategy": "no",
        "report_to": [],
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "gradient_checkpointing": True,
        "bf16": torch.cuda.is_bf16_supported(),
        "fp16": not torch.cuda.is_bf16_supported(),
        "remove_unused_columns": False,
    }
    if args.mode == "gspo":
        config_kwargs["importance_sampling_level"] = "sequence"
    else:
        config_kwargs["importance_sampling_level"] = "token"

    grpo_config = GRPOConfig(**filter_kwargs(GRPOConfig.__init__, config_kwargs))
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": grpo_config,
        "reward_funcs": [exact_arithmetic_reward],
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
    }
    trainer = GRPOTrainer(**filter_kwargs(GRPOTrainer.__init__, trainer_kwargs))
    trainer.add_callback(
        CSVStepMetricsCallback(
            output_file=args.metrics_file,
            estimate=ThroughputEstimate(
                prompt_tokens=args.max_prompt_length,
                completion_tokens=args.max_completion_length,
                num_generations=args.num_generations,
                train_batch_size=args.batch_size,
            ),
        )
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final_adapter"))


if __name__ == "__main__":
    main()
