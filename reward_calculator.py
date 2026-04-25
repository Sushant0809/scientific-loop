import json
import re
from typing import Optional


def compute_format_reward(code: str) -> float:
    """
    Cheap format-only reward — no code execution required.
    Creates gradient signal even when all completions fail to run,
    breaking the dead-batch problem (zero reward variance in a batch).

    Returns a value in roughly [0.0, 1.2].
    """
    r = 0.0

    # Syntax-valid Python: cheaply catches mangled outputs
    try:
        compile(code, "<string>", "exec")
        r += 0.4
    except SyntaxError:
        r -= 0.2

    # Has a METRICS: print statement at all
    if "METRICS:" in code:
        r += 0.3

    # Has valid JSON dict in the METRICS line
    m = re.search(r"METRICS:\s*(\{[^}]+\})", code)
    if m:
        try:
            json.loads(m.group(1))
            r += 0.3
        except (json.JSONDecodeError, ValueError):
            pass

    # Reasonable implementation length (100–4000 chars)
    length = len(code)
    if 100 <= length <= 4000:
        r += 0.2

    return round(r, 4)


def compute_step_reward(
    reproduction_score: float,
    prev_reproduction_score: float,
    execution_status: str,        # "success" | "error" | "timeout" | "blocked"
    step_number: int,
    current_code: str,
    previous_code: Optional[str],
    metrics_matched_count: int,
    total_metrics: int,
) -> float:
    reward = 0.0

    # 1. Core metric proximity (0-10 range)
    reward += reproduction_score * 10.0

    # 2. Improvement bonus — reward moving in the right direction
    if reproduction_score > prev_reproduction_score:
        improvement = reproduction_score - prev_reproduction_score
        reward += improvement * 5.0

    # 3. Length penalty — aggressively penalise trivially short completions.
    # A real Python implementation of a paper needs ≥80 characters.
    # Without this, the model collapses to 2-token outputs (which run silently, no error).
    code_len = len(current_code.strip())
    if code_len < 80:
        # Scales from -3.0 at 0 chars to 0.0 at 80 chars
        reward -= 3.0 * max(0.0, 1.0 - code_len / 80.0)

    # 4. Execution quality
    if execution_status == "error":
        reward -= 2.0
    elif execution_status == "timeout":
        reward -= 1.5
    elif execution_status == "blocked":
        reward -= 4.0
    elif execution_status == "success" and reproduction_score == 0:
        reward -= 0.5  # ran but produced nothing measurable

    # 5. Partial metric match bonus
    if total_metrics > 0:
        reward += (metrics_matched_count / total_metrics) * 2.0

    # 6. Efficiency pressure — discourages dragging episodes out
    reward -= 0.1 * step_number

    # 7. Anti-stagnation — penalize submitting identical code
    if previous_code and current_code.strip() == previous_code.strip():
        reward -= 3.0

    # Clip to prevent extreme outliers from destabilizing GRPO advantage estimates
    return round(max(-5.0, min(reward, 12.0)), 4)


def compute_terminal_reward(
    final_score: float,
    success_threshold: float = 0.80,
    code_ran: bool = False,
) -> float:
    if final_score >= success_threshold:
        return 20.0
    elif final_score >= 0.60:
        return 10.0
    elif final_score >= 0.40:
        return 5.0
    elif code_ran:
        return 2.0
    else:
        return -5.0
