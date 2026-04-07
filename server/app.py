# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Wildfire Environment.
"""
import os
from typing import List
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import WildfireAction, WildfireObservation, TaskResponse, GraderResponse, BaselineResponse
    from .wildfireEnvironment_environment import WildfireEnv
    from .tasks import TASKS, TaskGrader
    from .agent import BaselineAgent
except ModuleNotFoundError:
    from models import WildfireAction, WildfireObservation, TaskResponse, GraderResponse, BaselineResponse
    from server.wildfireEnvironment_environment import WildfireEnv
    from server.tasks import TASKS, TaskGrader
    from server.agent import BaselineAgent


# Create the app with web interface and README integration
app = create_app(
    WildfireEnv,
    WildfireAction,
    WildfireObservation,
    env_name="wildfireEnvironment",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# -----------------------------------------------------------------------------
# Additional Wildfire Routes & Integrations
# -----------------------------------------------------------------------------

# Mount static frontend
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def serve_frontend():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Wildfire ICS OpenEnv API is running. Visit /docs for the API."}

@app.get("/tasks", response_model=List[TaskResponse])
def list_tasks():
    return [
        {"name": t.name, "difficulty": t.difficulty, "description": t.description}
        for t in TASKS
    ]

def get_active_env():
    """Helper to try to find the currently active OpenEnv session instance."""
    if hasattr(app.state, "env_manager"):
        # Access the private dict of sessions to get the running environment
        envs = getattr(app.state.env_manager, "_envs", {})
        if envs:
            return list(envs.values())[-1]
    return WildfireEnv()

@app.get("/grader", response_model=GraderResponse)
def get_grader_scores():
    active_env = get_active_env()
    grader = TaskGrader(active_env)
    return {
        "task_scores": {
            "task_1": grader.grade_task_1(),
            "task_2": grader.grade_task_2(),
            "task_3": grader.grade_task_3(),
        }
    }

@app.get("/baseline", response_model=BaselineResponse)
def run_baseline():
    """Run the rule-based baseline agent on a fresh local env instance."""
    fresh_env = WildfireEnv()
    
    current_task = 1
    try:
        active_env = get_active_env()
        if hasattr(active_env, "task"):
            current_task = active_env.task
    except Exception:
        pass

    fresh_env.reset(task=current_task)
    agent = BaselineAgent(fresh_env)
    
    total_reward = 0.0
    steps = 0

    while not fresh_env.done:
        obs_obj = fresh_env._get_obs()
        obs_dict = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else obs_obj.dict()
        action_dict = agent.select_action(obs_dict)
        
        try:
            from ..models import WildfireAction
        except ModuleNotFoundError:
            from models import WildfireAction
            
        action_obj = WildfireAction(**action_dict)
        obs_res = fresh_env.step(action_obj)
        
        total_reward += obs_res.reward
        steps += 1

    grader = TaskGrader(fresh_env)
    return {
        "task_scores": {
            "task_1": grader.grade_task_1(),
            "task_2": grader.grade_task_2(),
            "task_3": grader.grade_task_3(),
        },
        "total_reward": round(total_reward, 4),
        "steps_taken": steps,
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
