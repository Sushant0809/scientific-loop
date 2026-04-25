"""
Local smoke test for the full training pipeline.
Verifies: model loads → generates code → env executes → reward computed → GRPO step.

Run from parent directory:
    cd "/Volumes/D Drive/Scalar-Hackathon"
    TOTAL_EPISODES=4 MAX_STEPS=2 MODEL_NAME=Qwen/Qwen2.5-Coder-0.5B-Instruct python3 ScientificLoop/test_training.py
"""
import os, sys
os.environ.setdefault("TOTAL_EPISODES", "4")
os.environ.setdefault("MAX_STEPS", "2")
os.environ.setdefault("MODEL_NAME", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
os.environ.setdefault("OUTPUT_DIR", "/tmp/scientific-loop-test")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from ScientificLoop import ScientificLoopAction, ScientificLoopEnv
from ScientificLoop.paper_corpus import format_paper_for_agent, sample_paper

ENV_URL    = os.environ.get("ENV_URL", "https://sushant0809-scientific-loop.hf.space")
MODEL_NAME = os.environ["MODEL_NAME"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
MAX_STEPS  = int(os.environ["MAX_STEPS"])
TOTAL_EPISODES = int(os.environ["TOTAL_EPISODES"])

os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are an ML engineer. Implement the methodology below as a Python script. "
    "Output ONLY Python code. Last line must be:\n"
    'METRICS: {"metric_name": value}'
)

print("=" * 60)
print("ScientificLoop Training Smoke Test")
print(f"  Model:    {MODEL_NAME}")
print(f"  Env URL:  {ENV_URL}")
print(f"  Episodes: {TOTAL_EPISODES}")
print(f"  Steps:    {MAX_STEPS}")
print("=" * 60)

# ── Step 1: Load model ────────────────────────────────────────────────────────
print("\n[1/4] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # CPU-safe
    device_map="cpu",
)
print(f"  Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

# ── Step 2: Connect to environment ───────────────────────────────────────────
print("\n[2/4] Connecting to environment...")
env = ScientificLoopEnv(base_url=ENV_URL).sync().connect()
test_reset = env.reset()
print(f"  Connected! Paper: {test_reset.observation.paper_title[:50]}...")

# ── Step 3: Manual generation test (model → code → reward) ───────────────────
print("\n[3/4] Manual generation test (model generates code, env scores it)...")
paper = sample_paper(episode_number=0)
prompt = f"{SYSTEM_PROMPT}\n\n{format_paper_for_agent(paper)}"
print(f"  Paper: {paper.paper_id}")

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
generated_code = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"  Generated ({len(generated_code)} chars): {generated_code[:100].strip()}...")

env.reset()
result = env.step(ScientificLoopAction(code=generated_code, reasoning="smoke test"))
print(f"  Reward: {result.reward:.2f} | Score: {result.observation.reproduction_score:.3f}")
print(f"  Achieved metrics: {result.observation.achieved_metrics}")
if result.observation.error_message:
    print(f"  Error: {result.observation.error_message[:100]}")

# ── Step 4: Full GRPO training step ──────────────────────────────────────────
print("\n[4/4] Running GRPO training (2 steps)...")

def reward_fn(prompts: list, completions: list, **kwargs) -> list[float]:
    rewards = []
    for code in completions:
        env.reset()
        r = env.step(ScientificLoopAction(code=code, reasoning=""))
        rew = float(r.reward or 0.0)
        rewards.append(rew)
        print(f"    reward={rew:.2f} | score={r.observation.reproduction_score:.3f} | achieved={r.observation.achieved_metrics}")
    return rewards

dataset = Dataset.from_list([
    {"prompt": f"{SYSTEM_PROMPT}\n\n{format_paper_for_agent(sample_paper(ep))}", "episode": ep}
    for ep in range(TOTAL_EPISODES)
])

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=2,  # must be >= num_generations
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    max_completion_length=200,
    num_generations=2,              # minimum for GRPO
    temperature=0.8,
    logging_steps=1,
    report_to="none",
    bf16=False,                     # CPU-safe
    save_strategy="no",
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=grpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()

print("\n" + "=" * 60)
print("SMOKE TEST PASSED — Full pipeline verified:")
print("  model generates code → env executes → reward returned → GRPO updates weights")
print("=" * 60)

env.close()
