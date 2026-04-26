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

**Trained model:** [Sushant0809/scientific-loop-grpo](https://huggingface.co/Sushant0809/scientific-loop-grpo) (Qwen2.5-Coder-7B + LoRA r=8, GRPO)

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
total_reward = exec_reward + 0.5 * format_reward

exec_reward = (
    reproduction_score * 10      # 0–10 scale metric proximity
  + improvement_bonus * 5        # reward moving in right direction
  - length_penalty               # penalize trivially short code (<80 chars)
  - execution_penalty            # -2 error, -1.5 timeout, -4 blocked
  + metric_match_bonus           # partial credit per matched metric
  - 0.1 * step_number            # efficiency pressure
)

format_reward = syntax_check + metrics_line + valid_json + length_check  # 0–1.2
```

The **format reward** is the key innovation — it scores code quality without executing it, ensuring non-zero reward variance even when all completions fail at runtime. This eliminates dead batches (zero GRPO gradient).

### Execution Sandbox

Code runs in an isolated subprocess with:
- No network access (blocked imports: `requests`, `urllib`, `socket`)
- No filesystem writes outside `/tmp`
- Per-paper CPU timeouts (10–55 seconds)
- Security error detection for blocked syscalls

---

## Training

We fine-tuned **Qwen/Qwen2.5-Coder-7B-Instruct** using **GRPO** (Group Relative Policy Optimization) via TRL 1.x with LoRA adapters.

### Setup

- **Base model:** Qwen2.5-Coder-7B-Instruct (7B parameters)
- **Fine-tuning:** LoRA r=8, α=16 (~80M trainable params, ~0.4% of model)
- **Algorithm:** GRPO — 8 completions per prompt, advantage = (reward − mean) / std
- **Hardware:** HuggingFace Jobs H200 (141GB VRAM)
- **Episodes:** 200 (curriculum: warmup → easy → medium → hard)
- **Epochs:** 1 (100 gradient steps, 1h 31min)
- **Max completion length:** 1024 tokens
- **Format reward:** Syntax check + METRICS line detection — eliminates dead batches

### Training Command

```bash
hf jobs uv run \
    --with "trl>=1.2.0" --with torch --with transformers \
    --with accelerate --with datasets --with peft --with matplotlib \
    --with "openenv-scientific-loop @ git+https://huggingface.co/spaces/Sushant0809/scientific-loop" \
    --flavor h200 -s HF_TOKEN \
    -- python train_grpo.py
```

### Key Design Choices

**Why LoRA?** Full fine-tuning of 7B parameters requires ~56GB VRAM for gradients + optimizer states. LoRA r=8 reduces this to ~1GB, making the entire training fit on any GPU with the model.

**Why 1024 token completions?** Our initial runs used 200 tokens — this caused 80–100% of completions to be truncated mid-code, resulting in SyntaxErrors and zero gradient signal. Increasing to 1024 tokens dropped the truncation rate to 20–40%.

**Why format reward?** Without it, 90% of GRPO batches had zero reward variance (all completions failed identically → same reward → zero gradient). The format reward provides signal without code execution, dropping dead batches from 90% to **0%**.

---

## Results

### Training Curves

![Training Curves](https://huggingface.co/Sushant0809/scientific-loop-grpo/resolve/main/training_curves.png)

**What the curves show:**

- **Mean Reward** (top-left): Starts at −2.2, 15 steps escape the −2.1 error floor, best reward = **+0.184** at step 42.
- **Reward Std** (top-right): 12 high-signal spikes (std > 3) — batches where some completions ran successfully and others didn't, giving GRPO maximum learning signal.
- **Dead Batches** (bottom-left): **0% dead batches across all 100 steps** — format reward eliminated this problem completely.
- **Escaped Floor** (bottom-right): 15 steps above −1.5 reward, showing the model learned to produce runnable code with measurable outputs.

### Training Statistics

| Metric | Value |
|--------|-------|
| Total steps | 100 |
| Runtime | 1h 31min |
| Best reward | **+0.184** (step 42) |
| Dead batches | **0%** (all 100 steps) |
| High-signal spikes (std > 3) | 12 |
| Steps above error floor | 15 |

### What the Model Learned

Even without perfect paper reproduction, the model demonstrably learned:
- To write syntactically valid Python (format reward gradient)
- To include `METRICS: {...}` output lines
- To write longer, more complete implementations (mean completion length grew from ~590 to ~860 tokens)
- To sometimes run code that produces measurable results (reward escaping the −2.1 floor)

---

## Quick Start

```python
import asyncio
from ScientificLoop import ScientificLoopEnv, ScientificLoopAction

async def main():
    async with ScientificLoopEnv(base_url="https://Sushant0809-scientific-loop.hf.space") as env:
        result = await env.reset()
        print(f"Paper: {result.observation.paper_title}")

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
├── reward_calculator.py         # Step reward + format reward
├── client.py                    # EnvClient WebSocket wrapper
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml               # Dependencies
├── train_grpo.py                # GRPO training script (TRL 1.x)
├── train_grpo.ipynb             # Colab training notebook
├── hf_blog_post.md              # Blog post
├── training_curves.png          # Training metrics visualization
└── server/
    ├── app.py                   # FastAPI server via OpenEnv create_app
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
```

---

## Links

- **Environment (HF Space):** https://huggingface.co/spaces/Sushant0809/scientific-loop
- **Trained Model (LoRA adapter):** https://huggingface.co/Sushant0809/scientific-loop-grpo
- **Training Job Logs:** https://huggingface.co/jobs/Sushant0809/69ed86d7d70108f37acdf6c5

---

*OpenEnv Hackathon — PyTorch × Scaler — April 2026*
