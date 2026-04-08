# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wildfire Environment Client."""

from typing import Dict, Any
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import WildfireAction, WildfireObservation

class WildfireClient(EnvClient[WildfireAction, WildfireObservation, State]):
    """
    Client for the Wildfire Environment.
    Maintains connection to the server for inference actions.
    """

    def _step_payload(self, action: WildfireAction) -> Dict:
        return {
            "division_id": action.division_id,
            "crew_id": action.crew_id,
            "action_type": action.action_type,
            "target_cell": action.target_cell,
        }

    def _parse_result(self, payload: Dict) -> StepResult[WildfireObservation]:
        obs_data = payload.get("observation", {})
        
        # Build strict dict wrapper
        observation = WildfireObservation(
            grid=obs_data.get("grid", []),
            wind=obs_data.get("wind", {}),
            crew_positions=obs_data.get("crew_positions", []),
            division_zones=obs_data.get("division_zones", []),
            assets_at_risk=obs_data.get("assets_at_risk", []),
            timestep=obs_data.get("timestep", 0),
            containment_pct=obs_data.get("containment_pct", 0),
            metadata=obs_data.get("metadata", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
