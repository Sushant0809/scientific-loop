import uuid
from typing import Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import ScientificLoopAction, ScientificLoopObservation, ScientificLoopState
    from ..paper_corpus import Paper, format_paper_for_agent, sample_paper
    from ..reward_calculator import compute_step_reward, compute_terminal_reward
    from .execution_engine import compute_metric_proximity, extract_metrics, run_code
except ImportError:
    from models import ScientificLoopAction, ScientificLoopObservation, ScientificLoopState
    from paper_corpus import Paper, format_paper_for_agent, sample_paper
    from reward_calculator import compute_step_reward, compute_terminal_reward
    from server.execution_engine import compute_metric_proximity, extract_metrics, run_code


class ScientificLoopEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 10
    SUCCESS_THRESHOLD: float = 0.80

    def __init__(self):
        super().__init__()
        self._episode_counter: int = 0
        self._paper: Optional[Paper] = None
        self._prev_code: Optional[str] = None
        self._prev_score: float = 0.0
        self._best_score: float = 0.0
        self._total_reward: float = 0.0
        self._step_count: int = 0
        self._step_history: list = []
        self._state = ScientificLoopState(
            episode_id="init",
            paper_id="none",
            step_count=0,
            max_steps=self.MAX_STEPS,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> ScientificLoopObservation:
        self._episode_counter += 1
        self._paper = sample_paper(episode_number=self._episode_counter)
        self._prev_code = None
        self._prev_score = 0.0
        self._best_score = 0.0
        self._total_reward = 0.0
        self._step_count = 0
        self._step_history = []

        ep_id = episode_id or str(uuid.uuid4())
        self._state = ScientificLoopState(
            episode_id=ep_id,
            paper_id=self._paper.paper_id,
            step_count=0,
            max_steps=self.MAX_STEPS,
            best_score=0.0,
            total_reward=0.0,
            status="running",
        )

        return ScientificLoopObservation(
            paper_title=self._paper.title,
            paper_abstract=self._paper.abstract,
            paper_methodology=format_paper_for_agent(self._paper),
            target_metrics=self._paper.target_metrics,
            last_code="",
            execution_output="",
            error_message="",
            achieved_metrics={},
            metric_delta={},
            reproduction_score=0.0,
            step_history=[],
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: ScientificLoopAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> ScientificLoopObservation:
        if self._paper is None:
            self.reset()

        self._step_count += 1
        self._state.step_count = self._step_count

        # Execute code in sandbox
        exec_timeout = int(timeout_s) if timeout_s else self._paper.execution_timeout
        stdout, stderr, timed_out = run_code(action.code, timeout=exec_timeout)

        # Determine execution status
        if "SecurityError" in stderr:
            exec_status = "blocked"
        elif timed_out:
            exec_status = "timeout"
        elif stderr and not stdout:
            exec_status = "error"
        else:
            exec_status = "success"

        # Extract and score metrics
        achieved_metrics = extract_metrics(stdout)
        repro_score, metric_delta = compute_metric_proximity(
            achieved=achieved_metrics,
            target=self._paper.target_metrics,
            weights=self._paper.metric_weights,
        )

        # Count matched metrics (within 10% tolerance)
        matched = sum(
            1 for m, tv in self._paper.target_metrics.items()
            if m in achieved_metrics
            and abs(achieved_metrics[m] - tv) / max(abs(tv), 1e-6) <= 0.10
        )

        # Compute step reward
        step_reward = compute_step_reward(
            reproduction_score=repro_score,
            prev_reproduction_score=self._prev_score,
            execution_status=exec_status,
            step_number=self._step_count,
            current_code=action.code,
            previous_code=self._prev_code,
            metrics_matched_count=matched,
            total_metrics=len(self._paper.target_metrics),
        )

        # Check termination
        done = (
            self._step_count >= self.MAX_STEPS
            or repro_score >= self.SUCCESS_THRESHOLD
        )

        # Add terminal reward on episode end
        if done:
            step_reward += compute_terminal_reward(
                final_score=repro_score,
                success_threshold=self.SUCCESS_THRESHOLD,
                code_ran=(exec_status == "success"),
            )

        self._total_reward += step_reward
        self._best_score = max(self._best_score, repro_score)
        self._state.best_score = self._best_score
        self._state.total_reward = self._total_reward
        if done:
            self._state.status = "success" if repro_score >= self.SUCCESS_THRESHOLD else "failed"

        # Record history and advance state
        self._step_history.append({
            "step": self._step_count,
            "score": repro_score,
            "status": exec_status,
            "reward": step_reward,
        })
        self._prev_code = action.code
        self._prev_score = repro_score

        return ScientificLoopObservation(
            paper_title=self._paper.title,
            paper_abstract=self._paper.abstract,
            paper_methodology=format_paper_for_agent(self._paper),
            target_metrics=self._paper.target_metrics,
            last_code=action.code,
            execution_output=stdout[:2000],
            error_message=stderr[:500],
            achieved_metrics=achieved_metrics,
            metric_delta=metric_delta,
            reproduction_score=repro_score,
            step_history=list(self._step_history),
            done=done,
            reward=step_reward,
        )

    @property
    def state(self) -> ScientificLoopState:
        return self._state
