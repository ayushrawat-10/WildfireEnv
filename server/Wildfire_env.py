# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import copy
from uuid import uuid4
from typing import Dict, List, Any, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import WildfireAction, WildfireObservation
except ImportError:
    from models import WildfireAction, WildfireObservation

# Cell states
CLEAR = 0
BURNING = 1
BURNT = 2
FIREBREAK = 3

# Division zones on 20x20 grid
DIVISIONS = {
    "A": {"rows": (0, 9),  "cols": (0, 9)},
    "B": {"rows": (0, 9),  "cols": (10, 19)},
    "C": {"rows": (10, 19),"cols": (0, 9)},
    "D": {"rows": (10, 19),"cols": (10, 19)},
}
DIV_INDEX = ["A", "B", "C", "D"]

WIND_DIRS = ["N", "S", "E", "W"]


class Wildfire_env(Environment):
    """
    Wildfire Incident Command RL Environment.
    Simulates ICS-based wildfire suppression on a 20x20 grid
    with hierarchical agent structure (Division Supervisors + Crew Bosses).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.grid_size = 20
        self.reset()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, task: int = 1) -> WildfireObservation:
        """Reset environment to starting state for a given task (1/2/3)."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task = task
        self.timestep = 0
        self.done = False
        self.grid = [[CLEAR] * self.grid_size for _ in range(self.grid_size)]
        self.assets_burned_this_step = []
        self.front_merges = 0

        # Wind
        if task == 3:
            rng = random.Random(42)  # Fixed seed for consistent wind in task 3
            self.wind = {
                "direction": rng.choice(WIND_DIRS),
                "speed": round(rng.uniform(0.3, 0.7), 2)
            }
        elif task == 2:
            self.wind = {"direction": "S", "speed": 0.5}
        else:
            self.wind = {"direction": "E", "speed": 0.3}

        # Max steps per task
        self.max_steps = {1: 30, 2: 50, 3: 60}[task]

        # Place initial fire
        self._init_fire(task)

        # Place assets
        self._init_assets(task)

        # Place crews (2 per division)
        self._init_crews()

        return self._get_obs(reward=0.0, done=False, info={"task": task})

    def step(self, action: WildfireAction) -> WildfireObservation:  # type: ignore[override]
        """Apply action, spread fire, compute reward."""
        if self.done:
            return self._get_obs(reward=0.0, done=True, info={"error": "Episode done"})

        self.assets_burned_this_step = []
        reward = 0.0
        info = {}

        old_containment = self._containment_pct()

        # Apply action
        action_dict = action.model_dump()
        result = self._apply_action(action_dict)
        info.update(result)

        # Spread fire
        self._spread_fire()

        # Check assets
        for asset in self.assets:
            if asset["status"] == "safe":
                r, c = asset["cell"]
                if self.grid[r][c] == BURNING or self.grid[r][c] == BURNT:
                    asset["status"] = "burned"
                    self.assets_burned_this_step.append(asset["id"])

        # Check front merges (task 3)
        if self.task == 3:
            merges = self._detect_merges()
            self.front_merges += merges
        else:
            merges = 0

        new_containment = self._containment_pct()
        containment_delta = new_containment - old_containment

        # Reward components
        assets_safe = sum(1 for a in self.assets if a["status"] == "safe")
        asset_ratio = assets_safe / len(self.assets) if self.assets else 1.0
        crew_eff = result.get("efficiency", 0.0)
        idle_pen = 1.0 if result.get("idle", False) else 0.0
        assets_destroyed = len(self.assets_burned_this_step)

        reward = (
            0.4 * containment_delta
            + 0.3 * asset_ratio
            + 0.2 * crew_eff
            - 0.1 * idle_pen
            - 0.5 * assets_destroyed
            - 0.8 * merges
        )

        self.timestep += 1
        self._state.step_count = self.timestep

        if self.timestep >= self.max_steps or self._all_fire_contained():
            self.done = True

        info["containment_pct"] = new_containment
        info["assets_safe"] = assets_safe
        
        return self._get_obs(reward=round(reward, 4), done=self.done, info=info)

    @property
    def state(self) -> State:
        """
        Get the current environment state for OpenEnv internal tracking.
        """
        return self._state

    def get_internal_state(self) -> Dict[str, Any]:
        """Full internal state snapshot for graders."""
        return {
            "grid": self.grid,
            "crews": self.crews,
            "wind": self.wind,
            "timestep": self.timestep,
            "containment_pct": self._containment_pct(),
            "assets": self.assets,
            "task": self.task,
            "done": self.done,
            "front_merges": self.front_merges,
        }

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self, reward: float = 0.0, done: bool = False, info: Dict[str, Any] = None) -> WildfireObservation:
        info = info or {}
        division_zones = []
        for i, (div_id, zone) in enumerate(DIVISIONS.items()):
            division_zones.append({
                "id": i,
                "label": div_id,
                "row_start": zone["rows"][0],
                "row_end": zone["rows"][1],
                "col_start": zone["cols"][0],
                "col_end": zone["cols"][1],
            })

        crew_positions = []
        for crew in self.crews:
            crew_positions.append({
                "id": crew["id"],
                "division": crew["division"],
                "cell": crew["cell"],
                "status": crew["status"],
            })

        burning_per_division = {}
        for div_label, zone in DIVISIONS.items():
            burning_per_division[div_label] = sum(
                1 for r in range(zone["rows"][0], zone["rows"][1] + 1)
                for c in range(zone["cols"][0], zone["cols"][1] + 1)
                if self.grid[r][c] == BURNING
            )

        return WildfireObservation(
            grid=self.grid,
            division_zones=division_zones,
            crew_positions=crew_positions,
            wind=self.wind,
            assets_at_risk=self.assets,
            timestep=self.timestep,
            containment_pct=round(self._containment_pct(), 4),
            burning_per_division=burning_per_division,
            reward=reward,
            done=done,
            metadata=info,
        )

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_fire(self, task: int):
        if task == 1:
            for r in range(2, 7):
                for c in range(2, 7):
                    self.grid[r][c] = BURNING
        elif task == 2:
            for c in range(5, 15):
                self.grid[0][c] = BURNING
        elif task == 3:
            for r in range(2, 6):
                self.grid[r][2] = BURNING
                self.grid[r][3] = BURNING
            for r in range(2, 6):
                self.grid[r][12] = BURNING
                self.grid[r][13] = BURNING
            for r in range(12, 16):
                self.grid[r][2] = BURNING
                self.grid[r][3] = BURNING

    def _init_assets(self, task: int):
        if task == 1:
            self.assets = []
        elif task == 2:
            self.assets = [
                {"id": "town",         "name": "Town",          "cell": [15, 10], "status": "safe"},
                {"id": "water_tower",  "name": "Water Tower",   "cell": [12, 5],  "status": "safe"},
                {"id": "road_junction","name": "Road Junction", "cell": [14, 15], "status": "safe"},
            ]
        elif task == 3:
            self.assets = [
                {"id": "A1", "name": "Asset A1", "cell": [7, 7],   "status": "safe"},
                {"id": "A2", "name": "Asset A2", "cell": [8, 2],   "status": "safe"},
                {"id": "B1", "name": "Asset B1", "cell": [7, 17],  "status": "safe"},
                {"id": "B2", "name": "Asset B2", "cell": [8, 12],  "status": "safe"},
                {"id": "C1", "name": "Asset C1", "cell": [17, 7],  "status": "safe"},
                {"id": "C2", "name": "Asset C2", "cell": [18, 2],  "status": "safe"},
            ]

    def _init_crews(self):
        self.crews = []
        crew_id = 0
        for div_idx, (div_label, zone) in enumerate(DIVISIONS.items()):
            r_mid = (zone["rows"][0] + zone["rows"][1]) // 2
            c_mid = (zone["cols"][0] + zone["cols"][1]) // 2
            for offset in [0, 1]:
                self.crews.append({
                    "id": crew_id,
                    "division": div_idx,
                    "division_label": div_label,
                    "cell": [r_mid, c_mid + offset],
                    "status": "active",
                })
                crew_id += 1

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------

    def _apply_action(self, action: Dict) -> Dict:
        try:
            div_id = int(action.get("division_id", 0))
            crew_id = int(action.get("crew_id", 0))
            action_type = str(action.get("action_type", "HOLD")).upper()
            target = action.get("target_cell", [0, 0])
            r, c = int(target[0]), int(target[1])
        except Exception:
            return {"error": "Invalid action format", "efficiency": 0.0, "idle": False}

        crew = self._find_crew(div_id, crew_id)
        if crew is None:
            return {"error": "Crew not found", "efficiency": 0.0, "idle": False}
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return {"error": "Target out of bounds", "efficiency": 0.0, "idle": False}

        efficiency = 0.0
        idle = False

        if action_type == "SUPPRESS":
            if self.grid[r][c] == BURNING:
                self.grid[r][c] = BURNT
                efficiency = 1.0
            crew["cell"] = [r, c]
        elif action_type == "FIREBREAK":
            if self.grid[r][c] == CLEAR:
                self.grid[r][c] = FIREBREAK
                efficiency = 0.5
            crew["cell"] = [r, c]
        elif action_type == "MOVE":
            crew["cell"] = [r, c]
        elif action_type == "HOLD":
            if self._fire_nearby(crew["cell"]):
                idle = True

        return {"action": action_type, "efficiency": efficiency, "idle": idle}

    def _find_crew(self, div_id: int, crew_id: int):
        div_crews = [c for c in self.crews if c["division"] == div_id]
        if crew_id < len(div_crews):
            return div_crews[crew_id]
        return None

    def _fire_nearby(self, cell: List[int], radius: int = 2) -> bool:
        r, c = cell
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if self.grid[nr][nc] == BURNING:
                        return True
        return False

    # ------------------------------------------------------------------
    # Fire spread
    # ------------------------------------------------------------------

    def _spread_fire(self):
        direction = self.wind["direction"]
        speed = self.wind["speed"]
        wind_offsets = {"N": [(-1, 0)], "S": [(1, 0)], "E": [(0, 1)], "W": [(0, -1)]}
        primary = wind_offsets[direction]
        all_neighbors = [(-1,0),(1,0),(0,-1),(0,1)]

        new_grid = copy.deepcopy(self.grid)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r][c] == BURNING:
                    if random.random() < 0.3:
                        new_grid[r][c] = BURNT
                    for dr, dc in all_neighbors:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                            continue
                        if self.grid[nr][nc] != CLEAR:
                            continue
                        prob = speed * 0.4
                        if (dr, dc) in primary:
                            prob = speed * 0.9
                        if random.random() < prob:
                            new_grid[nr][nc] = BURNING
        self.grid = new_grid

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _containment_pct(self) -> float:
        burning = sum(
            1 for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.grid[r][c] == BURNING
        )
        if burning == 0:
            return 1.0
        total_affected = sum(
            1 for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.grid[r][c] in (BURNING, BURNT)
        )
        if total_affected == 0:
            return 1.0
        return round(1.0 - (burning / max(total_affected, 1)), 4)

    def _all_fire_contained(self) -> bool:
        return all(
            self.grid[r][c] != BURNING
            for r in range(self.grid_size)
            for c in range(self.grid_size)
        )

    def _detect_merges(self) -> int:
        div_has_fire = {}
        for i, (label, zone) in enumerate(DIVISIONS.items()):
            div_has_fire[i] = any(
                self.grid[r][c] == BURNING
                for r in range(zone["rows"][0], zone["rows"][1] + 1)
                for c in range(zone["cols"][0], zone["cols"][1] + 1)
            )
        adjacent = [(0, 1), (0, 2), (1, 3), (2, 3)]
        return sum(1 for a, b in adjacent if div_has_fire.get(a) and div_has_fire.get(b))
