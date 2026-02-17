from __future__ import annotations

import re
from typing import Any

INT_PATTERN = re.compile(r"-?\d+")


def extract_first_int(text: str) -> str | None:
    match = INT_PATTERN.search(text)
    if match is None:
        return None
    return match.group(0)


def exact_arithmetic_reward(
    completions: list[Any],
    answer: list[str],
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion, target in zip(completions, answer):
        if isinstance(completion, list) and completion:
            content = completion[0].get("content", "")
        elif isinstance(completion, dict):
            content = completion.get("content", "")
        else:
            content = str(completion)
        predicted = extract_first_int(content.strip())
        rewards.append(1.0 if predicted == target else 0.0)
    return rewards
