# RL Feasibility Benchmark (1x 16GB GPU)

This project benchmarks `GRPO` and `GSPO` with `QLoRA` on Linux using two stacks:

- `Unsloth` runner (`scripts/run_unsloth_rl.py`)
- `TRL` runner (`scripts/run_trl_rl.py`)

Both use the same deterministic arithmetic task and the same CSV step metrics so results are comparable.

## What this measures

- Estimated tokens/sec per step
- Step time
- CUDA allocated/reserved/peak memory
- Basic training stability over short runs

The reward is exact integer match for arithmetic outputs (`benchmark/reward.py`) to avoid reward-model overhead.

## Environment (Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- Python `3.10+` is required for the Unsloth runner.
- Use a CUDA-enabled PyTorch build compatible with your local NVIDIA driver.
- If `bf16` is unavailable, scripts automatically switch to `fp16`.

## Single-run examples

Unsloth GRPO:

```bash
python scripts/run_unsloth_rl.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --mode grpo \
  --max-steps 200 \
  --batch-size 2 \
  --num-generations 4 \
  --output-dir outputs/unsloth_grpo
```

Unsloth GSPO:

```bash
python scripts/run_unsloth_rl.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --mode gspo \
  --max-steps 200 \
  --batch-size 2 \
  --num-generations 4 \
  --output-dir outputs/unsloth_gspo
```

TRL GRPO:

```bash
python scripts/run_trl_rl.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --mode grpo \
  --dtype fp16 \
  --max-steps 200 \
  --batch-size 2 \
  --num-generations 4 \
  --output-dir outputs/trl_grpo
```

TRL GSPO:

```bash
python scripts/run_trl_rl.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --mode gspo \
  --dtype fp16 \
  --max-steps 200 \
  --batch-size 2 \
  --num-generations 4 \
  --output-dir outputs/trl_gspo
```

## Full matrix

Run both frameworks, both modes, and sweep `K` and batch size:

```bash
python scripts/run_matrix.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --frameworks unsloth,trl \
  --modes grpo,gspo \
  --num-generations 2,4,8 \
  --batch-sizes 1,2 \
  --dtype fp16 \
  --max-steps 120
```

Summary table is written to:

- `outputs/matrix/summary.csv`

Per-run step metrics are written to:

- `outputs/matrix/<run_name>/metrics.csv`

## Analyze results

Rank and aggregate runs (including repeat-aware p10/median/p90 throughput):

```bash
python scripts/analyze_results.py \
  --input outputs/matrix/summary.csv \
  --output-dir outputs/matrix/analysis \
  --vram-cap-gb 15.5 \
  --top-n 20
```

Outputs:

- `outputs/matrix/analysis/ranked_configs.csv`
- `outputs/matrix/analysis/ranked_configs.md`

Optional charts:

```bash
pip install matplotlib
python scripts/analyze_results.py --plot
```

Plot files:

- `outputs/matrix/analysis/ranked_median_tps.png`
- `outputs/matrix/analysis/ranked_peak_vram.png`

## Practical defaults for 16GB 4070 Ti Super

- Start with `Qwen/Qwen2.5-3B-Instruct`
- `--max-prompt-length 256`
- `--max-completion-length 128`
- `--batch-size 1 or 2`
- `--num-generations 2 or 4` first, then try `8`

If you hit OOM:

- Lower `--num-generations` first
- Lower `--batch-size` second
- Lower prompt/completion lengths third
