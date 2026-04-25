"""
End-to-end test for ScientificLoop environment.
Run from the ScientificLoop parent directory:
    python3 -c "import asyncio; exec(open('ScientificLoop/test_environment.py').read())"
Or from inside the project:
    cd .. && python3 ScientificLoop/test_environment.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ScientificLoop import ScientificLoopAction, ScientificLoopEnv

BASE_URL = "http://localhost:8000"

WARMUP_CODE_LINEAR = """
import numpy as np
import json
np.random.seed(42)
x = np.linspace(-1, 1, 200)
y = 3 * x + 2 + np.random.normal(0, 0.1, 200)
w, b = 0.0, 0.0
for _ in range(500):
    yh = w * x + b
    e = yh - y
    w -= 0.1 * np.mean(e * x)
    b -= 0.1 * np.mean(e)
final_mse = float(np.mean((w * x + b - y) ** 2))
print(f"METRICS: {json.dumps({'final_mse': round(final_mse, 4), 'learned_w': round(w, 4), 'learned_b': round(b, 4)})}")
"""

BROKEN_CODE = "x = 1 + 1  # no METRICS print"


async def test_basic_episode():
    async with ScientificLoopEnv(base_url=BASE_URL) as env:
        result = await env.reset()
        obs = result.observation
        assert obs.paper_title, "Paper title should be set"
        assert obs.target_metrics, "Target metrics should be present"
        assert obs.paper_methodology, "Methodology prompt should be present"
        assert obs.done is False, "Should not be done after reset"
        print(f"[PASS] Reset: paper='{obs.paper_title[:50]}...'")

        result = await env.step(ScientificLoopAction(code=BROKEN_CODE, reasoning="broken code test"))
        obs = result.observation
        assert obs.reproduction_score == 0.0, "Broken code should give score 0"
        assert obs.achieved_metrics == {}, "Broken code should give no metrics"
        print(f"[PASS] Step with broken code: score={obs.reproduction_score}, reward={obs.reward:.2f}")

    print("[PASS] Basic episode test passed")


async def test_warmup_paper():
    """Run until we get the linear regression paper, then test correct implementation."""
    for attempt in range(10):
        async with ScientificLoopEnv(base_url=BASE_URL) as env:
            result = await env.reset()
            obs = result.observation
            if "Linear Regression" in obs.paper_title or "Gradient Descent" in obs.paper_title:
                result = await env.step(ScientificLoopAction(
                    code=WARMUP_CODE_LINEAR, reasoning="correct linear regression"
                ))
                obs = result.observation
                assert obs.reproduction_score > 0.5, f"Expected high score, got {obs.reproduction_score}"
                assert obs.reward > 0, f"Expected positive reward, got {obs.reward}"
                print(f"[PASS] Warmup paper test: score={obs.reproduction_score:.3f}, reward={obs.reward:.2f}")
                return
    print("[SKIP] Linear regression paper not sampled in 10 attempts (warmup curriculum working)")


async def test_max_steps_termination():
    """Verify episode terminates at MAX_STEPS."""
    async with ScientificLoopEnv(base_url=BASE_URL) as env:
        await env.reset()
        done_step = None
        for i in range(15):
            result = await env.step(ScientificLoopAction(code=BROKEN_CODE, reasoning=f"step {i+1}"))
            if result.observation.done:
                done_step = i + 1
                break
        assert done_step is not None, "Episode should have ended within 15 steps"
        assert done_step <= 10, f"Episode should end at MAX_STEPS=10, ended at {done_step}"
        print(f"[PASS] MAX_STEPS termination: episode ended at step {done_step}")


async def test_security_block():
    """Verify blocked imports are rejected."""
    from ScientificLoop.server.execution_engine import run_code
    malicious_code = "import requests\nprint('hacked')"
    stdout, stderr, timed_out = run_code(malicious_code, timeout=5)
    assert "SecurityError" in stderr, "Should block 'requests' import"
    print(f"[PASS] Security block: {stderr.strip()}")


async def run_all():
    print(f"Testing ScientificLoop environment at {BASE_URL}\n")
    await test_security_block()
    await test_basic_episode()
    await test_warmup_paper()
    await test_max_steps_termination()
    print("\n[ALL TESTS PASSED]")


if __name__ == "__main__":
    asyncio.run(run_all())
