from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ScientificLoopAction, ScientificLoopObservation


class ScientificLoopEnv(
    EnvClient[ScientificLoopAction, ScientificLoopObservation, State]
):
    """
    Client for the ScientificLoop environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with ScientificLoopEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.paper_title)
        ...     result = client.step(ScientificLoopAction(code="print('METRICS: {}')", reasoning="test"))
        ...     print(result.observation.reproduction_score)
    """

    def _step_payload(self, action: ScientificLoopAction) -> Dict:
        return {
            "code": action.code,
            "reasoning": action.reasoning or "",
        }

    def _parse_result(self, payload: Dict) -> StepResult[ScientificLoopObservation]:
        obs_data = payload.get("observation", {})
        valid_fields = ScientificLoopObservation.model_fields.keys()
        filtered = {k: v for k, v in obs_data.items() if k in valid_fields}
        observation = ScientificLoopObservation(
            **filtered,
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        valid_fields = State.model_fields.keys()
        filtered = {k: v for k, v in payload.items() if k in valid_fields}
        return State(**filtered)
