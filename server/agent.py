from typing import Dict, Any, List

BURNING = 1
CLEAR = 0

DIVISIONS = {
    0: {"rows": (0, 9),  "cols": (0, 9)},
    1: {"rows": (0, 9),  "cols": (10, 19)},
    2: {"rows": (10, 19),"cols": (0, 9)},
    3: {"rows": (10, 19),"cols": (10, 19)},
}


class BaselineAgent:
    """
    Rule-based baseline agent for the Wildfire ICS environment.

    Strategy:
    1. Find the division with the most burning cells.
    2. Direct both crew bosses in that division to SUPPRESS the nearest burning cell.
    3. If no burning cells in a division, HOLD.
    """

    def __init__(self, env):
        self.env = env

    def select_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        grid = obs["grid"]

        # Find division with most fire
        div_fire_counts = {}
        for div_id, zone in DIVISIONS.items():
            count = sum(
                1 for r in range(zone["rows"][0], zone["rows"][1] + 1)
                for c in range(zone["cols"][0], zone["cols"][1] + 1)
                if grid[r][c] == BURNING
            )
            div_fire_counts[div_id] = count

        hottest_div = max(div_fire_counts, key=lambda d: div_fire_counts[d])

        if div_fire_counts[hottest_div] == 0:
            # No fire anywhere — hold crew 0 of division 0
            return {
                "division_id": 0,
                "crew_id": 0,
                "action_type": "HOLD",
                "target_cell": [5, 5],
            }

        # Find nearest burning cell in hottest division
        zone = DIVISIONS[hottest_div]
        burning_cells = [
            (r, c)
            for r in range(zone["rows"][0], zone["rows"][1] + 1)
            for c in range(zone["cols"][0], zone["cols"][1] + 1)
            if grid[r][c] == BURNING
        ]

        # Pick boundary cell closest to other divisions (priority)
        target = burning_cells[0]
        if burning_cells:
            # Sort by position to prefer boundary cells
            target = sorted(burning_cells, key=lambda x: -(x[0] + x[1]))[0]

        # Alternate between crew 0 and crew 1 each step
        crew_id = self.env.timestep % 2

        return {
            "division_id": hottest_div,
            "crew_id": crew_id,
            "action_type": "SUPPRESS",
            "target_cell": list(target),
        }
