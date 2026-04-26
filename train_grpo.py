"""
GRPO training script for ScientificLoop — TRL 1.x compatible.

Key TRL 1.x changes vs 0.x:
  - tokenizer= is now processing_class=
  - max_new_tokens= is now max_completion_length=
  - reward_fn signature: (prompts, completions, **kwargs) -> list[float]
  - Dataset prompt field can be a plain string (non-conversational)
"""
import json
import os
import re
import sys

# Must be set before torch import to fix T4 memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import torch

# H200 sometimes reports CUDA Error 802 (system not yet initialized) on first probe.
# Retry for up to 3 minutes until the driver is ready before loading the model.
for _i in range(36):
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            torch.cuda.set_device(0)
            _ = torch.zeros(1, device="cuda")  # force full driver init
            print(f"CUDA ready: {torch.cuda.get_device_name(0)}")
            break
        except RuntimeError:
            pass
    print(f"Waiting for CUDA... attempt {_i+1}/36")
    time.sleep(5)
else:
    print("WARNING: CUDA not available after 3min, falling back to CPU")
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from ScientificLoop.paper_corpus import EVAL_PAPERS, format_paper_for_agent, load_paper, sample_paper
from ScientificLoop.server.execution_engine import compute_metric_proximity, extract_metrics, run_code
from ScientificLoop.reward_calculator import compute_format_reward, compute_step_reward, compute_terminal_reward

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")
ENV_URL     = os.environ.get("ENV_URL",    "https://sushant0809-scientific-loop.hf.space")
HF_FLAVOR   = os.environ.get("HF_FLAVOR", "a10g-large")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", "./outputs/scientific-loop-grpo")
TOTAL_EPISODES = int(os.environ.get("TOTAL_EPISODES", 200))
MAX_STEPS   = int(os.environ.get("MAX_STEPS", -1))   # -1 = use num_epochs instead

os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are an expert ML engineer. "
    "Read the paper methodology below and implement it as a complete, "
    "self-contained Python script. Output ONLY executable Python code — "
    "no markdown, no explanation, no code fences. "
    "The script must print its results on the last line as:\n"
    'METRICS: {"metric_name": value}'
)

# ── Model & Tokenizer ─────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ADAPTER_ID = os.environ.get("ADAPTER_ID", "Sushant0809/scientific-loop-grpo")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# Load previously trained LoRA adapter for warm-start (continued training).
# This picks up where epoch 1 left off instead of learning from scratch.
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID, is_trainable=True)
    print(f"Loaded LoRA adapter from {ADAPTER_ID} — warm-start training.")
    lora_config = None  # already a PEFT model, no new config needed
except Exception as e:
    print(f"Could not load adapter ({e}), falling back to fresh LoRA.")
    model = base_model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )

print(f"Trainable parameters with LoRA r=8:")

# No network env needed for training — we run code locally on the job machine
# (faster, no latency, evaluates against the CORRECT paper from the prompt)
print("Using local execution engine for reward computation.")


# ── Datasets ──────────────────────────────────────────────────────────────────
def make_dataset(total_episodes: int = TOTAL_EPISODES) -> Dataset:
    """Curriculum-ordered prompts. Each row = one GRPO episode."""
    rows = []
    for ep in range(total_episodes):
        paper = sample_paper(episode_number=ep)
        rows.append({
            "prompt": f"{SYSTEM_PROMPT}\n\n{format_paper_for_agent(paper)}",
            "paper_id": paper.paper_id,
            "episode": ep,
            "difficulty": paper.difficulty,
        })
    return Dataset.from_list(rows)


def make_eval_dataset() -> Dataset:
    return Dataset.from_list([
        {
            "prompt": f"{SYSTEM_PROMPT}\n\n{format_paper_for_agent(p)}",
            "paper_id": p.paper_id,
        }
        for p in EVAL_PAPERS
    ])


train_dataset = make_dataset()
eval_dataset  = make_eval_dataset()


# ── Reward function (TRL 1.x signature) ──────────────────────────────────────
def _extract_code(raw: str) -> str:
    """
    Extract Python code from a completion that may be wrapped in markdown fences
    or contain surrounding prose. Handles both ```python ... ``` and bare code.
    """
    raw = raw.strip()
    # Prefer an explicitly fenced block
    fence = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    # Remove any stray fence markers from instruct-model preamble/postamble
    raw = re.sub(r"```(?:python)?", "", raw)
    raw = re.sub(r"```", "", raw)
    return raw.strip()


def reward_fn(prompts: list, completions: list, **kwargs) -> list[float]:
    """
    TRL 1.x passes all dataset columns as **kwargs.
    We use kwargs["paper_id"] to evaluate each completion against the CORRECT paper.
    """
    paper_ids = kwargs.get("paper_id", [None] * len(completions))
    rewards = []
    for code, paper_id in zip(completions, paper_ids):
        paper = load_paper(paper_id)
        code = _extract_code(code)
        stdout, stderr, timed_out = run_code(code, paper.execution_timeout)

        if "SecurityError" in stderr: exec_status = "blocked"
        elif timed_out:               exec_status = "timeout"
        elif stderr and not stdout:   exec_status = "error"
        else:                         exec_status = "success"

        achieved = extract_metrics(stdout)
        score, _ = compute_metric_proximity(achieved, paper.target_metrics, paper.metric_weights)
        matched = sum(
            1 for m, tv in paper.target_metrics.items()
            if m in achieved and abs(achieved[m] - tv) / max(abs(tv), 1e-6) <= 0.10
        )
        exec_reward = compute_step_reward(
            reproduction_score=score,
            prev_reproduction_score=0.0,
            execution_status=exec_status,
            step_number=1,
            current_code=code,
            previous_code=None,
            metrics_matched_count=matched,
            total_metrics=len(paper.target_metrics),
        )
        fmt_reward = compute_format_reward(code)
        rewards.append(round(exec_reward + 0.5 * fmt_reward, 4))
    return rewards


# ── Eval callback ─────────────────────────────────────────────────────────────
class EvalReproductionCallback(TrainerCallback):
    def __init__(self, eval_every: int = 25):
        self.eval_every = eval_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every != 0:
            return
        m   = kwargs["model"]
        tok = kwargs.get("processing_class", tokenizer)
        scores = []
        for item in eval_dataset:
            inputs = tok(item["prompt"], return_tensors="pt").to(m.device)
            with torch.no_grad():
                out = m.generate(**inputs, max_new_tokens=512, temperature=0.2, do_sample=True)
            code = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            # Free eval generation memory before scoring
            del out, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            paper = load_paper(item["paper_id"])
            stdout, stderr, timed_out = run_code(code, paper.execution_timeout)
            achieved = extract_metrics(stdout)
            score, _ = compute_metric_proximity(achieved, paper.target_metrics, paper.metric_weights)
            scores.append(score)
        mean_score = sum(scores) / max(len(scores), 1)
        print(f"\n[Eval @ step {state.global_step}] Mean reproduction score: {mean_score:.3f}")
        with open(f"{OUTPUT_DIR}/eval_scores.jsonl", "a") as f:
            f.write(json.dumps({"step": state.global_step, "score": mean_score}) + "\n")
        # Clear GPU cache after full eval pass so checkpoint save has headroom
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── GRPO Config (TRL 1.x field names) ────────────────────────────────────────
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,                # 2 epochs = 200 steps ≈ 3hr on H200
    max_steps=MAX_STEPS,               # override epochs for quick local tests
    per_device_train_batch_size=8,     # increased from 4 — A100 80GB has headroom
    gradient_accumulation_steps=2,     # halved to keep effective batch size the same
    learning_rate=1e-5,
    max_completion_length=1024,
    num_generations=8,                 # increased from 4 — more diversity → stronger GRPO signal
    temperature=1.1,
    logging_steps=1,
    save_steps=50,          # mid-run checkpoint at step 50 (~1hr mark)
    save_total_limit=2,     # keep only last 2 checkpoints to save disk/memory
    report_to="none",
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=True,       # trades compute for memory — needed on T4
    push_to_hub=True,
    hub_model_id="Sushant0809/scientific-loop-grpo",
)

# ── Trainer (TRL 1.x: processing_class instead of tokenizer) ─────────────────
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=grpo_config,
    peft_config=lora_config,           # None when warm-starting (model is already PEFT)
    train_dataset=train_dataset,
    processing_class=tokenizer,
    callbacks=[EvalReproductionCallback(eval_every=25)],
)

print(f"\nStarting GRPO training — {TOTAL_EPISODES} episodes, model: {MODEL_NAME}")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ── Final eval ────────────────────────────────────────────────────────────────
print("\n=== FINAL EVALUATION ON HELD-OUT PAPERS ===")
for item in eval_dataset:
    inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True)
    code = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    paper = load_paper(item["paper_id"])
    stdout, stderr, timed_out = run_code(code, paper.execution_timeout)
    achieved = extract_metrics(stdout)
    score, _ = compute_metric_proximity(achieved, paper.target_metrics, paper.metric_weights)
    print(f"  {item['paper_id']:30s}  score={score:.3f}  achieved={achieved}")

print(f"\nModel saved to: {OUTPUT_DIR}")

# ── Training curves plot ───────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    history = trainer.state.log_history
    steps, rewards, stds, dead = [], [], [], []
    for entry in history:
        if "reward" in entry:
            steps.append(entry["step"])
            rewards.append(entry["reward"])
            stds.append(entry.get("reward_std", 0))
            dead.append(entry.get("frac_reward_zero_std", 0))

    eval_steps, eval_scores = [], []
    eval_path = f"{OUTPUT_DIR}/eval_scores.jsonl"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            for line in f:
                d = json.loads(line)
                eval_steps.append(d["step"])
                eval_scores.append(d["score"])

    def _smooth(v, w=5):
        if len(v) < w:
            return v
        return np.convolve(v, np.ones(w) / w, mode="valid").tolist()

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
    best = max(rewards) if rewards else 0
    fig.suptitle(
        f"ScientificLoop GRPO  |  {MODEL_NAME.split('/')[-1]} + LoRA r=8\n"
        f"{len(steps)} steps  ·  best reward: {best:.3f}",
        fontsize=13, fontweight="bold",
    )

    # Mean reward
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, rewards, color="#aecbfa", alpha=0.4, linewidth=1, label="raw")
    sm = _smooth(rewards)
    ax1.plot(steps[len(steps)-len(sm):], sm, color="#1a73e8", linewidth=2.5, label="smoothed (w=5)")
    ax1.axhline(-2.1, color="#d93025", linestyle="--", linewidth=1.2, label="error floor")
    ax1.fill_between(steps, rewards, -2.1, where=[r > -2.1 for r in rewards],
                     alpha=0.15, color="#34a853", label="above floor")
    ax1.set_title("Mean Reward per Step", fontweight="bold")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Reward")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # Reward std
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(steps, stds, color="#fbbc04", alpha=0.5, width=0.8, label="raw")
    sm2 = _smooth(stds)
    ax2.plot(steps[len(steps)-len(sm2):], sm2, color="#e37400", linewidth=2, label="smoothed")
    ax2.set_title("Reward Std  (>0 = gradient signal)", fontweight="bold")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Std")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Dead batches
    ax3 = fig.add_subplot(gs[1, 0])
    colors = ["#34a853" if v == 0 else ("#fbbc04" if v < 1 else "#ea4335") for v in dead]
    ax3.bar(steps, dead, color=colors, alpha=0.8, width=0.8)
    ax3.axhline(1.0, color="#d93025", linestyle="--", linewidth=1, label="all dead")
    pct = 100 * np.mean([v > 0 for v in dead]) if dead else 0
    ax3.set_title(f"Dead Batches  ({pct:.0f}% had zero std)", fontweight="bold")
    ax3.set_xlabel("Step"); ax3.set_ylabel("Fraction")
    ax3.set_ylim(0, 1.15); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # Eval score
    ax4 = fig.add_subplot(gs[1, 1])
    if eval_steps:
        ax4.plot(eval_steps, eval_scores, color="#a142f4", linewidth=2.5,
                 marker="o", markersize=7, label="eval repro score")
        ax4.axhline(0.8, color="#34a853", linestyle=":", linewidth=1.5, label="target (0.80)")
        for s, v in zip(eval_steps, eval_scores):
            ax4.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                         xytext=(0, 8), fontsize=8, color="#a142f4", ha="center")
    ax4.set_title("Eval Reproduction Score", fontweight="bold")
    ax4.set_xlabel("Step"); ax4.set_ylabel("Score (0–1)")
    ax4.set_ylim(-0.02, 1.05); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    plot_path = f"{OUTPUT_DIR}/training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to: {plot_path}")

    # Push plot to Hub alongside the model
    from huggingface_hub import HfApi
    HfApi().upload_file(
        path_or_fileobj=plot_path,
        path_in_repo="training_curves.png",
        repo_id="Sushant0809/scientific-loop-grpo",
        repo_type="model",
    )
    print("Training curves pushed to HuggingFace Hub.")
except Exception as e:
    print(f"Plot generation skipped: {e}")
