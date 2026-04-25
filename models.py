from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class ScientificLoopAction(Action):
    code: str = Field(..., description="Complete Python implementation to execute")
    reasoning: Optional[str] = Field(default="", description="Agent chain-of-thought (logged, not executed)")


class ScientificLoopObservation(Observation):
    # Static context — constant for entire episode
    paper_title: str = Field(default="", description="Title of the paper being reproduced")
    paper_abstract: str = Field(default="", description="Paper abstract")
    paper_methodology: str = Field(default="", description="Full prompt: methodology + task instructions")
    target_metrics: Dict[str, float] = Field(default_factory=dict, description="Ground truth metrics (server-side bookkeeping only)")

    # Dynamic context — changes each step
    last_code: str = Field(default="", description="Agent's previous code attempt")
    execution_output: str = Field(default="", description="stdout from last execution (capped 2000 chars)")
    error_message: str = Field(default="", description="stderr / exception text (capped 500 chars)")
    achieved_metrics: Dict[str, float] = Field(default_factory=dict, description="Metrics extracted from stdout")
    metric_delta: Dict[str, Optional[float]] = Field(default_factory=dict, description="achieved - target per metric")
    reproduction_score: float = Field(default=0.0, description="Weighted composite score 0.0–1.0")
    step_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of step results")
    # done and reward inherited from Observation base class


class ScientificLoopState(State):
    # episode_id and step_count inherited from State base class
    paper_id: str = Field(default="none", description="Which paper this episode is reproducing")
    max_steps: int = Field(default=10, description="Max steps per episode")
    best_score: float = Field(default=0.0, description="Best reproduction score achieved this episode")
    total_reward: float = Field(default=0.0, description="Cumulative reward this episode")
    status: str = Field(default="running", description="running | success | failed | timeout")
