from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from datasets import Dataset


SYSTEM_PROMPT = (
    "Solve arithmetic problems. Return only the final integer answer with no extra text."
)


@dataclass(frozen=True)
class ArithmeticSample:
    prompt: str
    answer: str


def make_arithmetic_samples(
    size: int,
    seed: int = 17,
    min_value: int = 10,
    max_value: int = 99,
) -> list[ArithmeticSample]:
    random.seed(seed)
    samples: list[ArithmeticSample] = []
    for _ in range(size):
        a = random.randint(min_value, max_value)
        b = random.randint(min_value, max_value)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question: What is {a} + {b}?\n"
            "Answer:"
        )
        samples.append(ArithmeticSample(prompt=prompt, answer=str(a + b)))
    return samples


def to_hf_dataset(samples: Iterable[ArithmeticSample]) -> Dataset:
    rows = [{"prompt": sample.prompt, "answer": sample.answer} for sample in samples]
    return Dataset.from_list(rows)
