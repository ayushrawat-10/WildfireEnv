# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Wildfire Environment.
"""

from typing import List, Dict, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class WildfireAction(Action):
    """Action for the Wildfire environment - commanding a crew."""

    division_id: int = Field(..., description="0=A, 1=B, 2=C, 3=D")
    crew_id: int = Field(..., description="0 or 1 within that division")
    action_type: str = Field(..., description="SUPPRESS | FIREBREAK | MOVE | HOLD")
    target_cell: List[int] = Field(..., description="[row, col]")


class WildfireObservation(Observation):
    """Observation from the Wildfire environment - full state footprint."""

    grid: List[List[int]] = Field(default_factory=list, description="20x20 grid of cell states: 0=CLEAR, 1=BURNING, 2=BURNT, 3=FIREBREAK")
    division_zones: List[Dict[str, Any]] = Field(default_factory=list, description="Boundaries of divisions A, B, C, D")
    crew_positions: List[Dict[str, Any]] = Field(default_factory=list, description="Positions and status of all crews")
    wind: Dict[str, Any] = Field(default_factory=dict, description="Wind direction and speed")
    assets_at_risk: List[Dict[str, Any]] = Field(default_factory=list, description="Critical assets dynamically checked")
    timestep: int = Field(default=0, description="Current simulation step")
    containment_pct: float = Field(default=0.0, description="Percentage of fire contained")
    burning_per_division: Dict[str, int] = Field(default_factory=dict, description="Number of burning cells by division")


from pydantic import BaseModel

class TaskResponse(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str
    grader: str

class GraderResponse(BaseModel):
    task_scores: Dict[str, float]

class BaselineResponse(BaseModel):
    task_scores: Dict[str, float]
    total_reward: float
    steps_taken: int

class ResetRequest(BaseModel):
    task: int = 1

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any] = {}

class StateResponse(BaseModel):
    state: Dict[str, Any]
