"""
GRPO training script for ScientificLoop.
Reference: https://huggingface.co/docs/trl/en/openenv
"""
import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from ScientificLoop import ScientificLoopAction, ScientificLoopEnv
from ScientificLoop.paper_corpus import EVAL_PAPERS, format_paper_for_agent, sample_paper

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ENV_URL = os.environ.get("ENV_URL", "https://Sushant0809-scientificloop.hf.space")
OUTPUT_DIR = "./outputs/scientific-loop-grpo"
TOTAL_EPISODES = int(os.environ.get("TOTAL_EPISODES", 200))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model & Tokenizer ────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ── Environment (sync client) ─────────────────────────────────────────────────
env = ScientificLoopEnv(base_url=ENV_URL).sync()

SYSTEM_PROMPT = (
    "You are an expert ML engineer. "
    "Read the paper methodology below and implement it as a complete, "
    "self-contained Python script. Output ONLY executable Python code -- "
    "no markdown, no explanation, no code fences. "
    "The script must print its measured results on the last line as: "
    'METRICS: {"metric_name": value}'
)


# ── Dataset: curriculum-ordered prompts ──────────────────────────────────────
def make_dataset(total_episodes: int = TOTAL_EPISODES) -> Dataset:
    prompts = []
    for ep in range(total_episodes):
        paper = sample_paper(episode_number=ep)
        prompt = f"{SYSTEM_PROMPT}\n\n{format_paper_for_agent(paper)}"
        prompts.append({
            "prompt": prompt,
            "paper_id": paper.paper_id,
            "episode": ep,
            "split": paper.split,
            "difficulty": paper.difficulty,
        })
    return Dataset.from_list(prompts)


def make_eval_dataset() -> Dataset:
    return Dataset.from_list([
        {
            "prompt": f"{SYSTEM_PROMPT}\n\n{format_paper_for_agent(p)}",
            "paper_id": p.paper_id,
            "split": "eval",
        }
        for p in EVAL_PAPERS
    ])


train_dataset = make_dataset()
eval_dataset = make_eval_dataset()

# ── Reward function ───────────────────────────────────────────────────────────
episode_counter = {"n": 0}


def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Called by TRL GRPO for each group of generated completions."""
    rewards = []
    for code, _prompt in zip(completions, prompts):
        env.reset()
        result = env.step(ScientificLoopAction(code=code, reasoning=""))
        rewards.append(float(result.reward or 0.0))
    episode_counter["n"] += len(completions)
    return rewards


# ── Eval callback ─────────────────────────────────────────────────────────────
class EvalReproductionCallback(TrainerCallback):
    def __init__(self, eval_every: int = 25):
        self.eval_every = eval_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every != 0:
            return
        model_obj = kwargs["model"]
        tok = kwargs.get("tokenizer", tokenizer)
        scores = []
        for item in eval_dataset:
            inputs = tok(item["prompt"], return_tensors="pt").to(model_obj.device)
            with torch.no_grad():
                out = model_obj.generate(**inputs, max_new_tokens=512, temperature=0.2, do_sample=True)
            code = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            env.reset()
            result = env.step(ScientificLoopAction(code=code, reasoning=""))
            scores.append(result.observation.reproduction_score)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        print(f"\n[Eval @ step {state.global_step}] Mean reproduction score: {mean_score:.3f}")
        with open(f"{OUTPUT_DIR}/eval_scores.jsonl", "a") as f:
            f.write(json.dumps({"step": state.global_step, "eval_repro_score": mean_score}) + "\n")


# ── GRPO Config ───────────────────────────────────────────────────────────────
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    max_new_tokens=512,
    num_generations=4,
    temperature=0.8,
    logging_steps=10,
    save_steps=50,
    report_to="none",
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    reward_funcs=reward_fn,
    tokenizer=tokenizer,
    callbacks=[EvalReproductionCallback(eval_every=25)],
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ── Final eval ────────────────────────────────────────────────────────────────
print("\n=== FINAL EVALUATION ON HELD-OUT PAPERS ===")
for item in eval_dataset:
    inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
    code = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    env.reset()
    result = env.step(ScientificLoopAction(code=code, reasoning=""))
    print(f"  {item['paper_id']:30s} → score: {result.observation.reproduction_score:.3f} | reward: {result.reward:.2f}")

print(f"\nModel saved to {OUTPUT_DIR}")
