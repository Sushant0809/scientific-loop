---
title: ScientificLoop Environment Server
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - scientific-reproducibility
---

# 🔬 ScientificLoop

> **Can an LLM learn to reproduce scientific papers through trial and error?**

ScientificLoop is an RL environment where an LLM agent reads an ML paper's methodology, generates Python code to implement it, executes that code in a sandbox, and receives reward proportional to how closely the measured results match the paper's reported values.

Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) for the PyTorch × Scaler OpenEnv Hackathon (April 2026).

**Trained model:** [Sushant0809/scientific-loop-grpo](https://huggingface.co/Sushant0809/scientific-loop-grpo) (Qwen2.5-Coder-7B + LoRA r=16, GRPO)

**Environment:** [Sushant0809/scientific-loop](https://huggingface.co/spaces/Sushant0809/scientific-loop)

**Training notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sushant0809/scientific-loop/blob/main/train_grpo.ipynb)

**Blog post:** [ScientificLoop: Teaching an LLM to Reproduce Scientific Papers with RL](https://huggingface.co/spaces/Sushant0809/scientific-loop/blob/main/hf_blog_post.md)

---

## The Problem

AI models can explain papers. They can summarize papers. But can they actually *reproduce* the results?

Reproducing a paper means writing code that, when executed, produces metrics within ~10% of the published values — not just code that looks right, but code that *runs right*. This is a hard, closed-loop task:

- The agent must understand the methodology precisely enough to implement it
- The implementation must be executable (no syntax errors, no import errors)
- The output format must be exact (`METRICS: {"metric_name": value}`)
- The computed values must match the paper's reported numbers

This is exactly the kind of task where RL can teach a model things that supervised fine-tuning cannot — because there is no "correct code" to imitate, only correct *outcomes*.

---

## Environment Design

### Episode Flow

```
Paper (title + abstract + methodology)
        ↓
   Agent generates Python code
        ↓
   Sandboxed subprocess executes code (CPU, timeout)
        ↓
   METRICS: {...} extracted from stdout
        ↓
   Reproduction score = metric proximity to paper targets
        ↓
   Reward signal → GRPO gradient update
```

Each episode = one paper. The agent has up to `MAX_STEPS=10` attempts per paper before the episode ends.

### Paper Corpus

18 ML papers across 3 curriculum levels — all designed to be reproducible using only standard libraries in under 55 seconds on CPU:

| Split | Count | Examples | Datasets |
|---|---|---|---|
| Warmup | 3 | Linear regression GD, Logistic regression, XOR MLP | Synthetic (numpy) |
| Train | 10 | Adam, Dropout, REINFORCE, VAE, DQN, BatchNorm, Word2Vec, PCA, Attention, ResNet | MNIST, CartPole, FrozenLake, synthetic |
| Eval | 5 | Nesterov SGD, Autoencoder, Q-learning, char-RNN, L2 regularization | MNIST, FrozenLake, synthetic |

Eval papers are **never sampled during training** — they test generalization.

### Reward Function

```python
step_reward = (
    reproduction_score * 10          # 0–10 scale metric proximity
  + improvement_bonus * 5            # reward moving in right direction
  - length_penalty                   # penalize trivially short code (<80 chars)
  - execution_penalty                # -2 error, -1.5 timeout, -4 blocked
  + metric_match_bonus               # partial credit per matched metric
  - 0.1 * step_number                # efficiency pressure
  - stagnation_penalty               # penalize identical re-submission
)

terminal_reward = 20 / 10 / 5 / 2 / -5  # full/strong/partial/ran/crashed
```

The reward is **dense** — every execution produces signal, not just successes.

### Metric Proximity

```python
score = Σ weight_i * exp(-|achieved_i - target_i| / (0.1 * |target_i|))
```

A metric is "matched" when it's within 10% of the target value. Score ranges from 0.0 (nothing right) to 1.0 (perfect reproduction).

### Execution Sandbox

Code runs in an isolated subprocess with:
- No network access (blocked imports: `requests`, `urllib`, `socket`)
- No filesystem writes outside `/tmp`
- Per-paper CPU timeouts (10–55 seconds)
- Security error detection for blocked syscalls

---

## Training

We fine-tuned **Qwen/Qwen2.5-Coder-7B-Instruct** using **GRPO** (Group Relative Policy Optimization) via TRL 1.x with LoRA adapters to fit in GPU memory.

### Setup

- **Base model:** Qwen2.5-Coder-7B-Instruct (7B parameters)
- **Fine-tuning:** LoRA r=16 on all linear layers (~160M trainable params, ~0.5% of model)
- **Algorithm:** GRPO — 4 completions per prompt, advantage = (reward − mean) / std
- **Hardware:** HuggingFace Jobs a10g-small (24GB VRAM)
- **Episodes:** 200 (curriculum: warmup → easy → medium → hard)
- **Epochs:** 2 (100 gradient steps, ~2.5hr)
- **Max completion length:** 1024 tokens
- **Format reward:** Syntax check + METRICS line detection (no execution) — eliminates dead batches

### Training Command

```bash
hf jobs uv run \
    --with "trl>=1.2.0" --with torch --with transformers \
    --with accelerate --with datasets --with peft \
    --with "openenv-scientific-loop @ git+https://huggingface.co/spaces/Sushant0809/scientific-loop" \
    --flavor a10g-small -s HF_TOKEN \
    -- python train_grpo.py
```

### Key Design Choices

**Why LoRA?** Full fine-tuning of 7B parameters requires ~56GB VRAM for gradients + optimizer states. LoRA reduces this to ~2GB, making the entire training fit comfortably on a 24GB a10g.

**Why 1024 token completions?** Our initial runs used 200 tokens — this caused 80–100% of completions to be truncated mid-code, resulting in SyntaxErrors on every completion and zero gradient signal. Increasing to 1024 tokens dropped the truncation rate to 20–40%, enabling real learning signal.

**Why local execution in the reward function?** Code runs directly on the training machine (no HTTP calls to the HF Space), giving faster reward computation and ensuring each completion is evaluated against the correct paper.

---

## Results

### Training Curves

![Training Curves](training_curves.png)

**What the curves show:**

- **Mean Reward** (top-left): Starts at −2.5 (erroring code), gradually rises above the −2.1 error floor. The spike at step 11 (`reward = −0.92, std = 4.74`) is the first batch where a completion actually ran and produced metrics.
- **Reward Diversity** (top-right): `reward_std > 0` on most steps means GRPO is computing non-zero gradients. The large spike at step 11 indicates high variance — one completion scored much higher than others.
- **Dead Batches** (bottom-left): Steps where `frac_reward_zero_std = 1` (all 4 completions got identical reward, zero gradient). Frequent but not dominant — the model is learning.
- **Completion Clip Ratio** (bottom-right): ~20–40% of completions hit the 1024-token limit (vs 80–100% with 200 tokens), showing the model writes longer, more complete code.

### Eval Scores

| Step | Mean Reproduction Score |
|------|------------------------|
| 25   | 0.000 |
| 50   | 0.000 |
| 75   | 0.000 |
| 100  | 0.000 |

The reproduction score on held-out eval papers remains 0 throughout training. This is expected given the task difficulty and limited compute (150 steps). The model is learning to generate runnable code (reward rising above −2.1 floor) but hasn't yet converged to generating code that produces metrics matching the exact target values.

### What the Model Learned

Even without hitting the reproduction score threshold, the model demonstrably learned:
- To write longer, more complete Python implementations (mean completion length: 617 → 800+ tokens)
- To generate code that actually executes (reward occasionally rises above −2.1 error floor)
- To produce `METRICS: {...}` formatted output (indicated by `exec_status = "success"` episodes)

Full convergence would require more training steps and potentially larger GPU compute (A100 × multiple runs).

---

## Quick Start

```python
import asyncio
from ScientificLoop import ScientificLoopEnv, ScientificLoopAction

async def main():
    async with ScientificLoopEnv(base_url="https://Sushant0809-scientific-loop.hf.space") as env:
        result = await env.reset()
        print(f"Paper: {result.observation.paper_title}")

        # Implement the paper methodology
        code = """
import numpy as np, json
np.random.seed(42)
x = np.linspace(-1, 1, 200)
y = 3*x + 2 + np.random.normal(0, 0.1, 200)
w, b = 0.0, 0.0
for _ in range(500):
    e = w*x + b - y
    w -= 0.1 * np.mean(e * x)
    b -= 0.1 * np.mean(e)
print(f"METRICS: {json.dumps({'final_mse': round(float(np.mean(e**2)), 4), 'learned_w': round(float(w), 4), 'learned_b': round(float(b), 4)})}")
"""
        result = await env.step(ScientificLoopAction(code=code))
        print(f"Score: {result.observation.reproduction_score:.3f} | Reward: {result.reward:.2f}")

asyncio.run(main())
```

---

## Project Structure

```
ScientificLoop/
├── __init__.py                  # Package exports
├── models.py                    # Pydantic Action/Observation/State
├── paper_corpus.py              # 18 papers + curriculum sampler
├── reward_calculator.py         # Step + terminal reward functions
├── client.py                    # EnvClient WebSocket wrapper
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml               # Dependencies
├── train_grpo.py                # GRPO training script (TRL 1.x)
├── training_curves.png          # Training metrics visualization
└── server/
    ├── app.py                   # FastAPI server via OpenEnv create_app
    ├── scientific_loop_environment.py  # Core RL env (reset/step/state)
    ├── execution_engine.py      # Sandboxed subprocess runner
    └── Dockerfile
```

---

## Local Development

```bash
# Install
pip install openenv-core torch torchvision numpy gymnasium

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Health check
curl http://localhost:8000/health

# Run a step
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"code": "import json\nprint(\"METRICS: {\\\"x\\\": 1.0}\")", "reasoning": "test"}'
```

---

## Links

- **Environment (HF Space):** https://huggingface.co/spaces/Sushant0809/scientific-loop
- **Trained Model (LoRA adapter):** https://huggingface.co/Sushant0809/scientific-loop-grpo
- **Training Job Logs:** https://huggingface.co/jobs/Sushant0809/69ece84bd70108f37acde9bd

---

*OpenEnv Hackathon — PyTorch × Scaler — April 2026*
