from typing import Optional


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

    # 3. Execution quality
    # NOTE: No bonus for merely running — only penalize failures.
    # A raw success bonus incentivises reward hacking (trivial 2-token code gets +1.9).
    # Positive signal comes entirely from metric proximity above.
    if execution_status == "error":
        reward -= 2.0
    elif execution_status == "timeout":
        reward -= 1.5
    elif execution_status == "blocked":
        reward -= 4.0
    elif execution_status == "success" and metrics_matched_count == 0 and reproduction_score == 0:
        reward -= 0.5  # ran but produced nothing useful

    # 4. Partial metric match bonus
    if total_metrics > 0:
        reward += (metrics_matched_count / total_metrics) * 2.0

    # 5. Efficiency pressure — discourages dragging episodes out
    reward -= 0.1 * step_number

    # 6. Anti-stagnation — penalize submitting identical code
    if previous_code and current_code.strip() == previous_code.strip():
        reward -= 3.0

    return round(reward, 4)


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
