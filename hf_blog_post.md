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

We designed 18 ML papers specifically for this task — not real papers with ImageNet results, but faithful algorithmic recreations using reproducible datasets (MNIST, CartPole, FrozenLake, synthetic data) that run in under 55 seconds on CPU:

| Split | Count | Examples |
|-------|-------|---------|
| Warmup | 3 | Linear regression GD, Logistic regression, XOR MLP |
| Train | 10 | Adam, Dropout, REINFORCE, VAE, DQN, BatchNorm, Word2Vec, PCA, Attention, ResNet |
| Eval (held out) | 5 | Nesterov SGD, Autoencoder, Q-learning, char-RNN, L2 regularization |

Eval papers are **never seen during training** — they test generalization.

---

## The Training

We fine-tuned **Qwen2.5-Coder-7B-Instruct** using **GRPO** (Group Relative Policy Optimization) with LoRA adapters on a HuggingFace Jobs a10g-small (24GB VRAM).

### Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| Base model | Qwen2.5-Coder-7B-Instruct | Strong coding baseline |
| LoRA rank | r=16, α=32 | ~160M trainable params — fits in 24GB |
| Episodes | 200 (curriculum) | Warmup → easy → medium → hard |
| GRPO generations | 4 per prompt | Diversity within memory budget |
| Max completion | 1024 tokens | Full Python implementations |
| Learning rate | 1e-5 | Standard for LoRA GRPO |
| Temperature | 1.1 | More diverse completions |

---

## What We Learned the Hard Way

### Problem 1: 80–100% Truncated Completions

Our initial runs used `max_completion_length=200`. This caused **most completions to be cut off mid-code**, resulting in SyntaxErrors on every completion → all 4 completions got identical reward → `reward_std = 0` → **zero gradient**.

**Fix:** Increased to `max_completion_length=1024`. Clipped ratio dropped from ~87% to 20–40%.

### Problem 2: 90% Dead Batches

Even with longer completions, our first working run showed `frac_reward_zero_std ≈ 0.9` — meaning **90% of GRPO batches had zero reward variance**, contributing no gradient update.

The root cause: when all 4 completions fail to execute (e.g., runtime error, missing METRICS line), they all get the same penalty → std = 0 → gradient = 0.

```
Old: frac_reward_zero_std = 0.9 → model barely learning
New: frac_reward_zero_std = 0.0 → real gradients every step
```

**Fix:** We added a `compute_format_reward` function that scores code quality **without running it** — using only static analysis:

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

This creates variance even when all 4 completions fail at runtime — because they still differ in syntax validity, presence of the METRICS line, and code length. The total reward becomes `exec_reward + 0.5 * format_reward`.

---

## Results

### Training Signal: Before vs After

The most important metric in GRPO is `frac_reward_zero_std` — the fraction of batches where all completions got the same reward (zero gradient). Here's what changed:

**Before (no format reward):**
- `frac_reward_zero_std` ≈ 0.9 (90% dead batches)
- Mean reward stuck at −2.1 floor throughout 100+ steps

**After (with format reward, steps 1–128):**
- `frac_reward_zero_std` = **0** on 95%+ of all steps
- Mean reward trajectory: −2.34 → −2.05 → −1.44 → **−0.61**

The high-variance spikes (`reward_std > 3.5`, occurring 8+ times) are particularly significant — they indicate batches where some completions successfully ran and produced measurable metrics while others didn't. These are the moments where GRPO has the clearest signal.

| Step range | Mean reward | Notable |
|-----------|-------------|---------|
| 1–10 | −2.1 to −2.3 | Warm-up, format learning |
| 19 | **−0.998** | First high-variance spike |
| 30 | **−1.394** | std = 3.91, strong signal |
| 89 | **−0.727** | Best in epoch 2 |
| 102 | **−0.609** | Best overall |

### What The Model Learned

Even without perfect paper reproduction, the model demonstrably learned:
- To write syntactically valid Python (syntax error rate dropped)
- To include `METRICS: {...}` output lines (format reward gradient)
- To write longer, more complete implementations (mean length: 550 → 860 tokens by step 120)
- To sometimes run code that produces measurable results (reward escaping the −2.1 floor)

Full numerical convergence (matching paper targets within 10%) would require more compute. But the learning signal is real and growing.

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

### Curriculum Sampling

```python
def sample_paper(episode_number):
    if episode_number <= 49:
        return random.choice(WARMUP_PAPERS)     # pure numpy, no downloads
    if episode_number <= 130:
        pool = easy*5 + medium*3 + hard*1       # weight easier papers
    else:
        pool = TRAIN_PAPERS                     # uniform
    return random.choice(pool)
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
        print(f"Task: {obs.observation.methodology[:200]}...")

        result = await env.step(ScientificLoopAction(code="<your implementation>"))
        print(f"Reproduction score: {result.observation.reproduction_score:.3f}")

asyncio.run(main())
```

**Training notebook:** [Open in Colab](https://colab.research.google.com/github/Sushant0809/scientific-loop/blob/main/train_grpo.ipynb)

**Trained model:** [Sushant0809/scientific-loop-grpo](https://huggingface.co/Sushant0809/scientific-loop-grpo) — Qwen2.5-Coder-7B + LoRA r=16, GRPO-trained

**Environment:** [Sushant0809/scientific-loop](https://huggingface.co/spaces/Sushant0809/scientific-loop)

---

*Built at the PyTorch × Scaler OpenEnv Hackathon, April 2026.*
