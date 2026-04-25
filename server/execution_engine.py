import json
import os
import re
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
from typing import Dict, Tuple


METRIC_PATTERN = re.compile(r"METRICS:\s*(\{.*?\})", re.DOTALL)

BLOCKED_IMPORTS = [
    "requests",
    "httpx",
    "urllib",
    "socket",
    "os.system",
    "os.popen",
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "shutil.rmtree",
]


def extract_metrics(stdout: str) -> Dict[str, float]:
    """Parse the agent's required METRICS: {...} output line."""
    matches = METRIC_PATTERN.findall(stdout)
    if not matches:
        return {}
    try:
        raw = json.loads(matches[-1])
        return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    except (json.JSONDecodeError, ValueError):
        return {}


def sanitize_code(code: str) -> Tuple[bool, str]:
    """Basic static check before execution. Returns (is_safe, reason)."""
    for blocked in BLOCKED_IMPORTS:
        # Check each line — skip if the match is inside a comment
        for line in code.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if blocked in line:
                return False, f"Blocked call detected: {blocked}"
    return True, ""


def run_code(code: str, timeout: int = 45) -> Tuple[str, str, bool]:
    """
    Execute agent code in an isolated subprocess.
    Returns: (stdout, stderr, timed_out)
    """
    is_safe, reason = sanitize_code(code)
    if not is_safe:
        return "", f"SecurityError: {reason}", False

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(textwrap.dedent(code))
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
            env={
                "PATH": os.path.dirname(sys.executable) + ":/usr/bin:/usr/local/bin",
                "HOME": "/tmp",
                # Include both venv and system site-packages so agent code can import
                # torch/gymnasium whether they are installed in venv or system-wide
                "PYTHONPATH": ":".join(filter(None, [
                    sysconfig.get_path("purelib"),
                    sysconfig.get_path("platlib"),
                    os.environ.get("PYTHONPATH", ""),
                ])),
            },
        )
        return result.stdout, result.stderr, False
    except subprocess.TimeoutExpired:
        return "", f"TimeoutError: execution exceeded {timeout}s", True
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def compute_metric_proximity(
    achieved: Dict[str, float],
    target: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (composite_score 0.0-1.0, per_metric_delta dict).
    Score of 1.0 = perfect reproduction. 0.0 = completely off or no metrics.
    """
    if not achieved or not target:
        return 0.0, {}

    total_weight = 0.0
    weighted_score = 0.0
    delta = {}

    for metric, target_val in target.items():
        if metric not in achieved:
            delta[metric] = None
            continue
        achieved_val = achieved[metric]
        weight = weights.get(metric, 1.0)

        if target_val == 0:
            pct_error = abs(achieved_val) / 1.0
        else:
            pct_error = abs(achieved_val - target_val) / abs(target_val)

        metric_score = max(0.0, 1.0 - pct_error)
        weighted_score += metric_score * weight
        total_weight += weight
        delta[metric] = achieved_val - target_val

    if total_weight == 0:
        return 0.0, delta

    return weighted_score / total_weight, delta
