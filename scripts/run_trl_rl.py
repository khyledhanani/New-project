#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

# Ensure project-root imports work regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data import make_arithmetic_samples, to_hf_dataset
from benchmark.metrics import CSVStepMetricsCallback, ThroughputEstimate
from benchmark.reward import exact_arithmetic_reward


def filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    allowed = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in allowed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL GRPO/GSPO QLoRA benchmark.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mode", choices=["grpo", "gspo"], default="grpo")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/trl"))
    parser.add_argument("--metrics-file", type=Path, default=Path("outputs/trl/metrics.csv"))
    parser.add_argument("--dataset-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16"],
        default="fp16",
        help="Compute dtype for quantized training. Use fp16 if bf16 causes type mismatch.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    return parser.parse_args()


def resolve_dtype(dtype_arg: str) -> tuple[torch.dtype, bool, bool]:
    if dtype_arg == "fp16":
        return torch.float16, False, True
    if dtype_arg == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("Requested --dtype bf16 but this CUDA setup does not support bf16.")
        return torch.bfloat16, True, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compute_dtype, use_bf16, use_fp16 = resolve_dtype(args.dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False

    train_dataset = to_hf_dataset(make_arithmetic_samples(size=args.dataset_size, seed=args.seed))
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

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
        "bf16": use_bf16,
        "fp16": use_fp16,
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
        "peft_config": lora_config,
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
