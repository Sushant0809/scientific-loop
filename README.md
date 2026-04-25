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

# ScientificLoop

**The first RL environment that trains AI to reproduce scientific papers — using the paper's own reported results as the reward signal.**

An OpenEnv-compliant RL environment where an LLM agent reads ML paper methodology, generates PyTorch code, executes it in a sandbox, and receives reward based on how closely the output metrics match the paper's reported values.

## Quick Start

```python
import asyncio
from ScientificLoop import ScientificLoopEnv, ScientificLoopAction

async def main():
    async with ScientificLoopEnv(base_url="https://Sushant0809-scientificloop.hf.space") as env:
        result = await env.reset()
        print(f"Paper: {result.observation.paper_title}")

        code = """
import numpy as np, json
np.random.seed(42)
x = np.linspace(-1,1,200); y = 3*x+2+np.random.normal(0,0.1,200)
w,b = 0.0,0.0
for _ in range(500):
    e=w*x+b-y; w-=0.1*np.mean(e*x); b-=0.1*np.mean(e)
print(f"METRICS: {json.dumps({'final_mse':round(float(np.mean(e**2)),4),'learned_w':round(w,4),'learned_b':round(b,4)})}")
"""
        result = await env.step(ScientificLoopAction(code=code, reasoning="linear regression impl"))
        print(f"Score: {result.observation.reproduction_score:.3f} | Reward: {result.reward:.2f}")

asyncio.run(main())
```

## Environment Design

### Episode Flow

1. Environment samples a paper from the 18-paper corpus (3 warmup / 10 train / 5 eval)
2. Agent receives paper title, abstract, and methodology as observation
3. Agent generates Python code as action
4. Code runs in a sandboxed subprocess with timeout
5. Metrics are extracted from the `METRICS: {...}` stdout line
6. Reward is computed based on proximity to paper's reported values
7. Episode ends when `MAX_STEPS=10` or `reproduction_score >= 0.80`

### Reward Function

```
step_reward = (score × 10) + (improvement × 5) + execution_bonus + match_bonus - efficiency_penalty
terminal_reward = 20 (full) / 10 (strong) / 5 (partial) / 2 (ran) / -5 (crashed)
```

### Paper Corpus

18 ML papers across 3 splits:

| Split | Count | Examples |
|---|---|---|
| Warmup | 3 | Linear regression, logistic regression, XOR MLP |
| Train | 10 | Adam, Dropout, REINFORCE, VAE, DQN, BatchNorm, Word2Vec, PCA, Attention, ResNet |
| Eval | 5 | Nesterov SGD, Autoencoder, Q-learning, char-RNN, L2 regularization |

### Metric Reporting Contract

Agent code **must** print metrics in this exact format:

```python
import json
print(f"METRICS: {json.dumps({'metric_name': value})}")
```

## Local Development

```bash
# Install dependencies
pip install openenv-core torch torchvision numpy gymnasium

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Run tests
python3 test_environment.py

# Health check
curl http://localhost:8000/health
```

## Training

```bash
# Set ENV_URL to your deployed space
export ENV_URL=https://Sushant0809-scientificloop.hf.space

# Local 5-episode smoke test
TOTAL_EPISODES=5 python3 train_grpo.py

# Full training on HF Jobs (T4 GPU)
hf jobs uv run \
    --with trl --with torch --with transformers --with gymnasium \
    --flavor t4-small -s HF_TOKEN \
    -- train_grpo.py
```

## Deployment

```bash
huggingface-cli login
openenv push --repo-id <your-username>/scientific-loop
```

## Project Structure

```
ScientificLoop/
├── __init__.py                         # Package exports
├── models.py                           # Pydantic Action/Observation/State
├── paper_corpus.py                     # 18 papers + curriculum sampler
├── reward_calculator.py                # Step + terminal reward functions
├── client.py                           # EnvClient WebSocket wrapper
├── openenv.yaml                        # OpenEnv manifest
├── pyproject.toml                      # Dependencies
├── train_grpo.py                       # GRPO training script
├── test_environment.py                 # End-to-end tests
└── server/
    ├── app.py                          # FastAPI server
    ├── ScientificLoop_environment.py   # Core RL environment
    ├── execution_engine.py             # Sandboxed subprocess runner
    └── Dockerfile
```

---

*OpenEnv Hackathon — April 2026 | Track: Multi-Agent Interactions / World Modeling*
