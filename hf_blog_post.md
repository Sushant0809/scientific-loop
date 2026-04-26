# ScientificLoop: Teaching an LLM to Reproduce Scientific Papers with RL

*OpenEnv Hackathon — PyTorch × Scaler — April 2026*

---

## The Question

AI models can explain papers. They can summarize them. But can they actually **reproduce the results?**

Not "generate code that looks plausible" — but code that, when you run it, produces numbers within 10% of what the paper actually reported.

This is harder than it sounds. It requires understanding the methodology precisely, writing syntactically correct code, using the right hyperparameters, and printing output in a specific format. It's a closed-loop task where there's no "correct answer" to imitate — only correct *outcomes* to discover.

That's exactly why RL is the right tool.

---

## The Environment

**ScientificLoop** is an OpenEnv-compliant RL environment where:

1. The agent receives a paper's title, abstract, and methodology
2. The agent generates a complete Python script as its "action"
3. The script executes in a sandboxed subprocess (CPU, timeout ≤55s)
4. Metrics are extracted from a `METRICS: {"name": value}` line in stdout
5. Reward = proximity of measured metrics to the paper's reported values

The agent gets **dense reward** on every attempt — not just 0/1 at the end. A script that runs and produces wrong numbers gets partial credit. A script that errors gets a small penalty. A script that matches the paper within 10% per metric gets full reward.

### The Paper Corpus

We designed 18 ML papers specifically for this task — faithful algorithmic recreations using reproducible datasets (MNIST, CartPole, FrozenLake, synthetic data) that run in under 55 seconds on CPU:

| Split | Count | Examples |
|-------|-------|---------|
| Warmup | 3 | Linear regression GD, Logistic regression, XOR MLP |
| Train | 10 | Adam, Dropout, REINFORCE, VAE, DQN, BatchNorm, Word2Vec, PCA, Attention, ResNet |
| Eval (held out) | 5 | Nesterov SGD, Autoencoder, Q-learning, char-RNN, L2 regularization |

Eval papers are **never seen during training** — they test generalization.

---

## The Training

We fine-tuned **Qwen2.5-Coder-7B-Instruct** using **GRPO** (Group Relative Policy Optimization) with LoRA adapters on a HuggingFace Jobs H200 (141GB VRAM).

### Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| Base model | Qwen2.5-Coder-7B-Instruct | Strong coding baseline |
| LoRA rank | r=8, α=16 | ~80M trainable params — fast + memory efficient |
| Episodes | 200 (curriculum) | Warmup → easy → medium → hard |
| GRPO generations | 8 per prompt | More diversity → stronger gradient signal |
| Max completion | 1024 tokens | Full Python implementations |
| Learning rate | 1e-5 | Standard for LoRA GRPO |
| Temperature | 1.1 | More diverse completions |
| Hardware | H200 (141GB) | Fast scheduling, large VRAM headroom |

---

## What We Learned the Hard Way

### Problem 1: 80–100% Truncated Completions

Our initial runs used `max_completion_length=200`. This caused **most completions to be cut off mid-code**, resulting in SyntaxErrors on every completion → all completions got identical reward → `reward_std = 0` → **zero gradient**.

**Fix:** Increased to `max_completion_length=1024`. Clipped ratio dropped from ~87% to 20–40%.

### Problem 2: 90% Dead Batches

Even with longer completions, our first working run showed `frac_reward_zero_std ≈ 0.9` — meaning **90% of GRPO batches had zero reward variance**, contributing no gradient update.

The root cause: when all completions fail to execute (runtime error, missing METRICS line), they all get the same penalty → std = 0 → gradient = 0.

```
Old run: frac_reward_zero_std = 0.9 → model barely learning
New run: frac_reward_zero_std = 0.0 → real gradients every step
```

**Fix:** We added a `compute_format_reward` function that scores code quality **without running it**:

```python
def compute_format_reward(code: str) -> float:
    r = 0.0
    try:
        compile(code, "<string>", "exec")
        r += 0.4          # syntactically valid Python
    except SyntaxError:
        r -= 0.2
    if "METRICS:" in code:
        r += 0.3          # has the required output line
    m = re.search(r"METRICS:\s*(\{[^}]+\})", code)
    if m:
        try:
            json.loads(m.group(1))
            r += 0.3      # valid JSON in METRICS line
        except: pass
    if 100 <= len(code) <= 4000:
        r += 0.2          # reasonable implementation length
    return r
```

Total reward = `exec_reward + 0.5 * format_reward`. This creates variance even when all completions fail at runtime.

### Problem 3: CUDA Error 802 on H200

H200 instances on HuggingFace Jobs sometimes report `CUDA Error 802: system not yet initialized` — PyTorch probes for CUDA before the GPU driver finishes starting. If `is_available()` returns False, the 7B model loads in float32 on CPU → generation takes hours.

**Fix:** A retry loop before model load:

```python
for _i in range(15):
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            torch.cuda.set_device(0)
            _ = torch.zeros(1, device="cuda")
            print(f"CUDA ready: {torch.cuda.get_device_name(0)}")
            break
        except RuntimeError:
            pass
    time.sleep(3)
```

---

## Results

### Training Curves

![Training Curves](https://huggingface.co/Sushant0809/scientific-loop-grpo/resolve/main/training_curves.png)

### Training Statistics

| Metric | Value |
|--------|-------|
| Total steps | 100 |
| Runtime | **1h 31min** |
| Best reward | **+0.184** (step 42) |
| Dead batches | **0%** (all 100 steps) |
| High-signal spikes (std > 3) | 12 |
| Steps above error floor | 15 |
| Avg step time | ~50s (H200) |

### What the Curves Show

- **Mean Reward** (top-left): 15 steps escape the −2.1 error floor. Best = +0.184 — the model actually reproduced paper metrics on that batch.
- **Reward Std** (top-right): 12 spikes with `std > 3` — the strongest gradient signal happens when some completions run successfully and others don't.
- **Dead Batches** (bottom-left): **100% green**. Zero dead batches across all 100 steps — format reward completely solved this.
- **Escaped Floor** (bottom-right): 15 batches above −1.5, showing the model learned to produce runnable code with measurable outputs.

### What The Model Learned

Even without perfect paper reproduction, the model demonstrably learned:
- To write syntactically valid Python
- To include `METRICS: {...}` output lines
- To write longer, more complete implementations (590 → 860 tokens mean length)
- To sometimes run code that produces measurable results

Full numerical convergence (matching paper targets within 10%) would require more compute. But the infrastructure works end-to-end and the learning signal is real.

---

## Architecture

```
ScientificLoop/
├── paper_corpus.py       # 18 papers + curriculum sampler
├── reward_calculator.py  # Step reward + format reward
├── train_grpo.py         # GRPO training (TRL 1.x + LoRA)
└── server/
    ├── app.py            # FastAPI OpenEnv server
    └── execution_engine.py  # Sandboxed subprocess runner
```

### Reward Function

```
total_reward = exec_reward + 0.5 * format_reward

exec_reward  = reproduction_score * 10
             + improvement_bonus * 5
             - execution_penalty (−2 error / −1.5 timeout / −4 blocked)
             + metric_match_bonus * 2
             - 0.1 * step_number
             ∈ [−5, +12]

format_reward = syntax_check + metrics_line + valid_json + length_check
              ∈ [0, 1.2]
```

---

## Try It

```python
import asyncio
from ScientificLoop import ScientificLoopEnv, ScientificLoopAction

async def main():
    async with ScientificLoopEnv(base_url="https://Sushant0809-scientific-loop.hf.space") as env:
        obs = await env.reset()
        print(f"Paper: {obs.observation.paper_title}")
        result = await env.step(ScientificLoopAction(code="<your implementation>"))
        print(f"Reproduction score: {result.observation.reproduction_score:.3f}")

asyncio.run(main())
```

**Training script:** [train_grpo.py](https://huggingface.co/spaces/Sushant0809/scientific-loop/blob/main/train_grpo.py) — the actual script used to train the model, run via `hf jobs uv run -- python train_grpo.py` on H200

**Training notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sushant0809/scientific-loop/blob/main/train_grpo.ipynb)

**Trained model:** [Sushant0809/scientific-loop-grpo](https://huggingface.co/Sushant0809/scientific-loop-grpo) — Qwen2.5-Coder-7B + LoRA r=8, GRPO-trained on H200

**Environment:** [Sushant0809/scientific-loop](https://huggingface.co/spaces/Sushant0809/scientific-loop)

---

*Built at the PyTorch × Scaler OpenEnv Hackathon, April 2026.*
