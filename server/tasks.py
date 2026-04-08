from typing import List

BURNING = 1
BURNT = 2

DIVISIONS = {
    "A": {"rows": (0, 9),  "cols": (0, 9)},
    "B": {"rows": (0, 9),  "cols": (10, 19)},
    "C": {"rows": (10, 19),"cols": (0, 9)},
    "D": {"rows": (10, 19),"cols": (10, 19)},
}


class Task:
    def __init__(self, id: str, name: str, difficulty: str, description: str, max_steps: int, grader: str):
        self.id = id
        self.name = name
        self.difficulty = difficulty
        self.description = description
        self.max_steps = max_steps
        self.grader = grader


class TaskGrader:
    """
    Deterministic grader for all 3 Wildfire tasks.
    Returns a score between 0.0 and 1.0.
    All graders use env.state() for evaluation.
    """

    def __init__(self, env):
        self.env = env

    def grade_task_1(self) -> float:
        """
        Task 1 (Easy): Contain fire within Division A.
        Score = (cells_in_A_contained / total_affected) * speed_bonus
        """
        state = self.env.get_internal_state()
        grid = state["grid"]
        zone = DIVISIONS["A"]

        total_burning = sum(
            1 for r in range(20) for c in range(20)
            if grid[r][c] == BURNING
        )

        # Fire outside Division A
        outside_burning = sum(
            1 for r in range(20) for c in range(20)
            if grid[r][c] == BURNING
            and not (zone["rows"][0] <= r <= zone["rows"][1]
                     and zone["cols"][0] <= c <= zone["cols"][1])
        )

        if total_burning == 0:
            containment_score = 1.0
        else:
            in_A = total_burning - outside_burning
            containment_score = in_A / total_burning

        # Speed bonus: full bonus if done before 20 steps
        timestep = state["timestep"]
        speed_bonus = max(0.5, 1.0 - (timestep / 30) * 0.5)

        return round(max(0.001, min(0.999, containment_score * speed_bonus)), 4)

    def grade_task_2(self) -> float:
        """
        Task 2 (Medium): Contain fire AND protect 3 assets.
        Score = containment_ratio * (assets_saved / 3)
        """
        state = self.env.get_internal_state()
        grid = state["grid"]

        total_cells = 20 * 20
        burning_cells = sum(
            1 for r in range(20) for c in range(20)
            if grid[r][c] == BURNING
        )
        containment_ratio = 1.0 - (burning_cells / total_cells)

        assets = state["assets"]
        assets_saved = sum(1 for a in assets if a["status"] == "safe")
        asset_score = assets_saved / 3 if assets else 1.0

        return round(max(0.001,min(0.999, containment_ratio * asset_score)), 4)

    def grade_task_3(self) -> float:
        """
        Task 3 (Hard): Contain all 3 fire fronts, protect 6 assets, no merges.
        Score = weighted_containment * speed_factor * asset_factor
        """
        state = self.env.get_internal_state()
        grid = state["grid"]
        timestep = state["timestep"]
        front_merges = state.get("front_merges", 0)

        # Per-division containment (A, B, C — D starts clear)
        div_scores = []
        for label in ["A", "B", "C"]:
            zone = DIVISIONS[label]
            burning = sum(
                1 for r in range(zone["rows"][0], zone["rows"][1] + 1)
                for c in range(zone["cols"][0], zone["cols"][1] + 1)
                if grid[r][c] == BURNING
            )
            total = 100  # 10x10
            div_scores.append(1.0 - (burning / total))

        avg_containment = sum(div_scores) / len(div_scores)

        # Speed factor
        speed_factor = max(0.5, 1.0 - (timestep / 60) * 0.5)

        # Asset factor
        assets = state["assets"]
        assets_saved = sum(1 for a in assets if a["status"] == "safe")
        asset_factor = assets_saved / len(assets) if assets else 1.0

        # Merge penalty
        merge_penalty = min(0.5, front_merges * 0.1)

        score = avg_containment * speed_factor * asset_factor - merge_penalty
        return round(max(0.001, min(0.999, score)), 4)


TASKS = [
    Task(
        "task_1",
        "Single Front Containment",
        "Easy",
        "Fire starts in Division A. Contain all fire within Division A within 30 steps.",
        30,
        "TaskGrader.grade_task_1()",
    ),
    Task(
        "task_2",
        "Asset Protection",
        "Medium",
        "Fire starts on north edge. Suppress fire AND keep Town, Water Tower, and Road Junction safe within 50 steps.",
        50,
        "TaskGrader.grade_task_2()",
    ),
    Task(
        "task_3",
        "Multi-Front Outbreak",
        "Hard",
        "Fire ignites in 3 divisions simultaneously. Coordinate all crews to contain all fronts within 75 steps.",
        75,
        "TaskGrader.grade_task_3()",
    ),
]
