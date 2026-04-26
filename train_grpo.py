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

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from ScientificLoop.paper_corpus import EVAL_PAPERS, format_paper_for_agent, load_paper, sample_paper
from ScientificLoop.server.execution_engine import compute_metric_proximity, extract_metrics, run_code
from ScientificLoop.reward_calculator import compute_format_reward, compute_step_reward, compute_terminal_reward

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")
ENV_URL     = os.environ.get("ENV_URL",    "https://sushant0809-scientific-loop.hf.space")
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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# LoRA keeps trainable params ~0.5% of model, making GRPO fit in 22GB VRAM.
# Base model (14GB bfloat16) + LoRA adapters + activations stays well under limit.
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
)
print(f"Trainable parameters with LoRA r=16:")

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
    Code runs locally on the job machine — no network calls, no random paper mismatch.

    Reward = execution_reward + 0.5 * format_reward
    The format_reward creates gradient signal even when all completions fail to run,
    preventing dead batches (zero reward variance → zero GRPO gradient).
    """
    paper_ids = kwargs.get("paper_id", [None] * len(completions))
    rewards = []
    for code, paper_id in zip(completions, paper_ids):
        paper = load_paper(paper_id)
        code = _extract_code(code)
        stdout, stderr, timed_out = run_code(code, paper.execution_timeout)

        if "SecurityError" in stderr:
            exec_status = "blocked"
        elif timed_out:
            exec_status = "timeout"
        elif stderr and not stdout:
            exec_status = "error"
        else:
            exec_status = "success"

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
        # Format reward provides gradient signal before the model learns to execute code.
        # Weighted at 0.5 so it doesn't overwhelm the execution quality signal.
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
    num_train_epochs=2,                # 3→2: 150 steps was hitting HF 4hr job timeout
    max_steps=MAX_STEPS,               # override epochs for quick local tests
    per_device_train_batch_size=4,     # must be >= num_generations
    gradient_accumulation_steps=4,
    learning_rate=1e-5,                # increased from 5e-6 — LoRA GRPO trains faster at 1e-5
    max_completion_length=1024,
    num_generations=4,                 # keep at 4 — T4 VRAM ceiling with 7B model
    temperature=1.1,                   # slightly higher → more diverse completions per prompt
    logging_steps=1,
    save_steps=75,          # offset from eval_every=25 so save never overlaps eval
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
    reward_funcs=reward_fn,            # TRL 1.x: positional arg 2
    args=grpo_config,
    peft_config=lora_config,           # LoRA: only adapter params are trained
    train_dataset=train_dataset,
    processing_class=tokenizer,        # TRL 1.x: was tokenizer=
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
