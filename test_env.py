"""
test_env.py — Smoke tests for the Wildfire ICS OpenEnv environment.
Run with: python test_env.py
"""

from server.Wildfire_env import Wildfire_env
from server.tasks import TaskGrader, TASKS
from server.agent import BaselineAgent
from models import WildfireAction


def test_reset_all_tasks():
    env = Wildfire_env()
    for task in [1, 2, 3]:
        obs_obj = env.reset(task=task)
        obs = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else obs_obj.dict()
        assert "grid" in obs, "grid missing from observation"
        assert len(obs["grid"]) == 20, "grid should be 20 rows"
        assert len(obs["grid"][0]) == 20, "grid should be 20 cols"
        assert "wind" in obs, "wind missing"
        assert "crew_positions" in obs, "crew_positions missing"
        assert "division_zones" in obs, "division_zones missing"
        assert len(obs["division_zones"]) == 4, "should have 4 divisions"
        assert len(obs["crew_positions"]) == 8, "should have 8 crew bosses"
        print(f"  ✓ reset(task={task}) passed")


def test_step_returns_correct_shape():
    env = Wildfire_env()
    env.reset(task=1)
    action_dict = {
        "division_id": 0,
        "crew_id": 0,
        "action_type": "SUPPRESS",
        "target_cell": [3, 3],
    }
    action = WildfireAction(**action_dict)
    obs = env.step(action)
    assert isinstance(obs.reward, float), "reward should be float"
    assert isinstance(obs.done, bool), "done should be bool"
    assert hasattr(obs, "grid"), "obs should have grid"
    print("  ✓ step() returns correct shape")


def test_all_action_types():
    env = Wildfire_env()
    env.reset(task=1)
    actions = [
        {"division_id": 0, "crew_id": 0, "action_type": "SUPPRESS",  "target_cell": [3, 3]},
        {"division_id": 0, "crew_id": 0, "action_type": "FIREBREAK", "target_cell": [0, 9]},
        {"division_id": 0, "crew_id": 1, "action_type": "MOVE",      "target_cell": [5, 5]},
        {"division_id": 1, "crew_id": 0, "action_type": "HOLD",      "target_cell": [5, 15]},
    ]
    for action_dict in actions:
        action = WildfireAction(**action_dict)
        obs = env.step(action)
        print(f"  ✓ action_type={action_dict['action_type']} reward={obs.reward:.4f}")


def test_grader_all_tasks():
    env = Wildfire_env()
    for task in [1, 2, 3]:
        env.reset(task=task)
        grader = TaskGrader(env)
        s1 = grader.grade_task_1()
        s2 = grader.grade_task_2()
        s3 = grader.grade_task_3()
        assert 0.0 < s1 < 1.0, f"task_1 score out of strict range: {s1}"
        assert 0.0 < s2 < 1.0, f"task_2 score out of strict range: {s2}"
        assert 0.0 < s3 < 1.0, f"task_3 score out of strict range: {s3}"
        print(f"  ✓ grader task={task} → t1={s1} t2={s2} t3={s3}")


def test_baseline_agent():
    env = Wildfire_env()
    for task in [1, 2, 3]:
        env.reset(task=task)
        agent = BaselineAgent(env)
        steps = 0
        total_reward = 0.0
        while not env.done:
            obs_obj = env._get_obs()
            obs = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else obs_obj.dict()
            action_dict = agent.select_action(obs)
            action = WildfireAction(**action_dict)
            obs_res = env.step(action)
            total_reward += obs_res.reward
            steps += 1
        grader = TaskGrader(env)
        score = [grader.grade_task_1(), grader.grade_task_2(), grader.grade_task_3()][task - 1]
        print(f"  ✓ baseline task={task} steps={steps} reward={total_reward:.2f} score={score}")


def test_state_method():
    env = Wildfire_env()
    env.reset(task=2)
    state = env.get_internal_state()
    required = ["grid", "crews", "wind", "timestep", "containment_pct", "assets", "task", "done"]
    for key in required:
        assert key in state, f"state() missing key: {key}"
    print("  ✓ get_internal_state() has all required keys")


if __name__ == "__main__":
    print("\n=== Wildfire ICS OpenEnv — Smoke Tests ===\n")

    print("▶ test_reset_all_tasks")
    test_reset_all_tasks()

    print("\n▶ test_step_returns_correct_shape")
    test_step_returns_correct_shape()

    print("\n▶ test_all_action_types")
    test_all_action_types()

    print("\n▶ test_grader_all_tasks")
    test_grader_all_tasks()

    print("\n▶ test_baseline_agent")
    test_baseline_agent()

    print("\n▶ test_state_method")
    test_state_method()

    print("\n=== All tests passed ✓ ===\n")
