"""
Inference Script Example (Wildfire ICS Agent)
===================================
Uses HuggingFace Inference Router (OpenAI-compatible API) to drive
the language model through wildfire suppression tasks, matching the
OpenEnv spec exactly.
"""

import asyncio
import os
import json
import argparse
from typing import List, Optional, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from client import WildfireClient
from models import WildfireAction

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
BENCHMARK = os.getenv("BENCHMARK", "wildfireEnvironment")
IMAGE_NAME = os.getenv("IMAGE_NAME") # If using docker image

MAX_STEPS = 15
SUCCESS_SCORE_THRESHOLD = 0.5


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_action_from_llm(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask the LLM to choose a fire suppression action based on the current observation.
    Returns a dict with division_id, crew_id, action_type, target_cell.
    """
    grid = obs.get("grid", [])
    division_summary = {}
    for zone_info in obs.get("division_zones", []):
        label = zone_info["label"]
        burning = sum(
            1 for r in range(zone_info["row_start"], zone_info["row_end"] + 1)
            for c in range(zone_info["col_start"], zone_info["col_end"] + 1)
            if grid[r][c] == 1
        )
        division_summary[label] = {"id": zone_info["id"], "burning_cells": burning}

    # Default fallback target
    busiest = max(division_summary.items(), key=lambda x: x[1]["burning_cells"])
    busiest_div_id = busiest[1]["id"]
    busiest_label = busiest[0]
    zone_meta = next(z for z in obs.get("division_zones", []) if z["label"] == busiest_label)
    target_cell = [zone_meta.get("row_start", 0) + 2, zone_meta.get("col_start", 0) + 2]

    found = False
    for r in range(zone_meta.get("row_start", 0), zone_meta.get("row_end", 0) + 1):
        for c in range(zone_meta.get("col_start", 0), zone_meta.get("col_end", 0) + 1):
            if grid and r < len(grid) and c < len(grid[0]) and grid[r][c] == 1:
                target_cell = [r, c]
                found = True
                break
        if found:
            break
            
    pct = obs.get("containment_pct", 0.0) * 100
    wind = obs.get("wind", {"direction": "NONE", "speed": 0})
    assets = obs.get("assets_at_risk", [])

    prompt = f"""You are an Incident Commander for a wildfire suppression operation.
Current timestep: {obs.get('timestep', 0)}
Containment: {pct:.1f}%
Wind: {wind['direction']} at speed {wind['speed']}

Division fire summary (burning cell counts):
{json.dumps(division_summary, indent=2)}

Assets at risk:
{json.dumps([{a['name']: a['status']} for a in assets], indent=2)}

Choose ONE action to direct a crew boss. Respond ONLY with a JSON object.

JSON Keys:
- "division_id": integer 0-3 (0=A, 1=B, 2=C, 3=D)
- "crew_id": integer 0 or 1 (two crews per division)
- "action_type": "SUPPRESS", "FIREBREAK", "MOVE", or "HOLD"
- "target_cell": [row, col] (integers 0-19)

Busiest division is {busiest_label} (id={busiest_div_id}) — suggested target: {target_cell}

Reply with ONLY the JSON object:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:].strip()
        text = text.strip()

        action = json.loads(text)
        for key in ["division_id", "crew_id", "action_type", "target_cell"]:
            if key not in action:
                raise ValueError(f"Missing key: {key}")
        return action

    except Exception as e:
        print(f"[DEBUG] LLM fallback due to {e}", flush=True)
        return {
            "division_id": busiest_div_id,
            "crew_id": 0,
            "action_type": "SUPPRESS",
            "target_cell": target_cell,
        }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM inference on Wildfire ICS OpenEnv")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3],
                        help="Task number (1=easy, 2=medium, 3=hard).")
    args = parser.parse_args()

    task = args.task
    task_name = {1: "single_front_containment", 2: "asset_protection", 3: "multi_front_outbreak"}.get(task, f"task_{task}")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        if IMAGE_NAME:
            env = await WildfireClient.from_docker_image(IMAGE_NAME)
            eval_base_url = getattr(env, "_base_url", "http://localhost:8000")
        else:
            # Fallback to local running instance
            eval_base_url = "http://localhost:8000"
            env = WildfireClient(base_url=eval_base_url)
            
        async with env:
            # The client reset usually lacks kwargs, but we want to specify task=task
            # We call the HTTP manually if `reset(task=x)` isn't properly supported on client.
            import httpx
            async with httpx.AsyncClient() as http:
                await http.post(f"{eval_base_url}/reset", json={"task": task})
            
            result = await env.reset()
            obs_obj = result.observation
            obs_dict = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else obs_obj.dict()
            
            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                action_dict = get_action_from_llm(client, obs_dict)
                action_str = json.dumps(action_dict).replace('"', "'") # remove internal double quotes for stdout
                
                result = await env.step(WildfireAction(**action_dict))
                
                obs_obj = result.observation
                obs_dict = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else obs_obj.dict()
                
                reward = result.reward or 0.0
                done = result.done
                error = None
                
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)
                
                if done:
                    break

            # Let's fetch score from our custom grader remotely
            async with httpx.AsyncClient() as http:
                resp = await http.get(f"{eval_base_url}/grader")
                if resp.status_code == 200:
                    scores = resp.json().get("task_scores", {})
                    score = scores.get(f"task_{task}", 0.0)
                else:
                    score = sum(rewards) / MAX_STEPS

            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[ERROR] Inference failed: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def run_inference(task: int = 1) -> dict:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    task_name = {1: "single_front_containment", 2: "asset_protection", 3: "multi_front_outbreak"}.get(task, f"task_{task}")

    try:
        if IMAGE_NAME:
            env = await WildfireClient.from_docker_image(IMAGE_NAME)
            eval_base_url = getattr(env, "_base_url", "http://localhost:8000")
        else:
            eval_base_url = "http://localhost:8000"
            env = WildfireClient(base_url=eval_base_url)

        async with env:
            import httpx
            async with httpx.AsyncClient() as http:
                await http.post(f"{eval_base_url}/reset", json={"task": task})

            result = await env.reset()
            obs_obj = result.observation
            obs_dict = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else obs_obj.dict()

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                action_dict = get_action_from_llm(client, obs_dict)
                result = await env.step(WildfireAction(**action_dict))

                obs_obj = result.observation
                obs_dict = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else obs_obj.dict()

                reward = result.reward or 0.0
                done = result.done

                rewards.append(reward)
                steps_taken = step

                if done:
                    break

            async with httpx.AsyncClient() as http:
                resp = await http.get(f"{eval_base_url}/grader")
                if resp.status_code == 200:
                    scores = resp.json().get("task_scores", {})
                    score = scores.get(f"task_{task}", 0.0)
                else:
                    score = sum(rewards) / MAX_STEPS

            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        return {"error": str(e)}

    return {
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards,
        "task": task_name
    }

if __name__ == "__main__":
    asyncio.run(main())