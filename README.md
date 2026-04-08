# 🔥 Wildfire Incident Command — OpenEnv

> A hierarchical multi-agent reinforcement learning environment simulating real-world wildfire suppression using the Incident Command System (ICS).

---

## 🌎 Environment Description & Motivation

Wildfires are one of the most destructive and fast-moving natural disasters, requiring coordinated decision-making across multiple teams operating simultaneously under extreme time pressure. Real agencies like **CAL FIRE** and **NWCG** use the **Incident Command System (ICS)** — a standardized hierarchical structure where an Incident Commander delegates to Division Supervisors, who direct Crew Bosses on the ground.

This environment simulates that exact structure on a dynamic **20×20 grid** where fire spreads each timestep based on wind direction and fuel probability. The goal is to train and evaluate agents that can:

- Reason hierarchically across multiple command levels
- Triage multiple simultaneous fire fronts
- Protect named civilian and infrastructure assets
- Coordinate crew actions across divisions under time pressure

**Why this matters:** Most RL environments test flat, single-agent control. This environment specifically targets *hierarchical multi-agent coordination* — a harder, more realistic problem with direct real-world applicability to emergency response, disaster management, and autonomous field operations.

---

## 🗺️ Simulation Overview

- **Grid size:** 20×20 cells divided into 4 Division zones (A, B, C, D)
- **Fire spread:** Each timestep, fire spreads to neighboring cells based on wind direction and a stochastic fuel probability model
- **Firebreak mechanic:** Agents can clear cells to block fire spread
- **2 Crew Bosses per division** (8 total across the map)

### Cell States

| Value | State | Description |
|-------|-------|-------------|
| 0 | Clear | Unaffected cell |
| 1 | Burning | Actively on fire — spreads each step |
| 2 | Burnt | Previously burned — no longer spreads |
| 3 | Firebreak | Cleared by crew — blocks fire spread |

### ICS Hierarchy

| Level | Role | Scope | Responsibility |
|-------|------|-------|----------------|
| Level 1 | Incident Commander | Entire map | Allocates divisions, sets overall strategy |
| Level 2 | Division Supervisor | One division zone | Directs crew bosses within their zone |
| Level 3 | Crew Boss | Local fire line | Direct suppression, firebreaks, movement |

### Division Layout

```
+----------+----------+
|          |          |
| Division | Division |
|    A     |    B     |
| rows 0-9 | rows 0-9 |
| cols 0-9 |cols 10-19|
+----------+----------+
|          |          |
| Division | Division |
|    C     |    D     |
|rows 10-19|rows 10-19|
| cols 0-9 |cols 10-19|
+----------+----------+
```

---

## 👁️ Observation Space

The observation is a dictionary returned after every `reset()` and `step()` call.

| Field | Type | Description |
|-------|------|-------------|
| `grid` | `List[List[int]]` | 20×20 fire state matrix (0=clear, 1=burning, 2=burnt, 3=firebreak) |
| `division_zones` | `List[Zone]` | Zone ID, label, row/col ranges for each division A–D |
| `crew_positions` | `List[Crew]` | Each crew: id, division, current cell, status |
| `wind` | `WindVector` | Direction (N/S/E/W) and speed (0.0–1.0) |
| `assets_at_risk` | `List[Asset]` | Named assets with id, name, cell location, status (safe/burned) |
| `timestep` | `int` | Current step number in the episode |
| `containment_pct` | `float` | Overall fire containment percentage (0.0–1.0) |
| `burning_per_division` | `Dict[str, int]` | Count of burning cells per division label |

---

## ⚡ Action Space

Each action directs **one crew boss** in **one division** to perform **one operation** on a target cell.

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `division_id` | `int` | 0–3 | Division to command: 0=A, 1=B, 2=C, 3=D |
| `crew_id` | `int` | 0–1 | Crew boss index within the division |
| `action_type` | `enum` | SUPPRESS, FIREBREAK, MOVE, HOLD | Operation to perform |
| `target_cell` | `[int, int]` | [0–19, 0–19] | Target grid coordinates [row, col] |

### Action Meanings

| Action | Effect | Efficiency |
|--------|--------|------------|
| `SUPPRESS` | Converts a BURNING cell to BURNT | 1.0 |
| `FIREBREAK` | Converts a CLEAR cell to FIREBREAK (blocks spread) | 0.5 |
| `MOVE` | Repositions the crew to a new cell | 0.0 |
| `HOLD` | Crew stays in place — penalized if fire is nearby | −0.1 |

---

## 🏆 Reward Function

Reward is calculated **every step** — not just at episode end — providing dense learning signal throughout the trajectory.

```
reward = (0.4 × containment_delta)
       + (0.3 × asset_safety_ratio)
       + (0.2 × crew_efficiency)
       - (0.1 × idle_penalty)
       - (0.5 × assets_destroyed)
       - (0.8 × front_merge)
```

| Component | Weight | Type | Trigger |
|-----------|--------|------|---------|
| `containment_delta` | +0.4 | Positive | Every step fire containment % increases |
| `asset_safety_ratio` | +0.3 | Positive | Every step based on assets remaining safe |
| `crew_efficiency` | +0.2 | Positive | When a crew action directly reduces burning cells |
| `idle_penalty` | −0.1 | Negative | When HOLD is used near active fire |
| `asset_destroyed` | −0.5 | Negative | Each time an asset is burned this step |
| `front_merge_penalty` | −0.8 | Negative | Task 3 only: two fire fronts merge across division boundary |

---

## 📋 Tasks

### Task 1 — Single Front Containment `[EASY]`

**Max steps:** 30 | **Wind:** Fixed East, speed 0.3

Fire starts as a **5×5 cluster** in the center of Division A. The agent must contain all fire within Division A — no fire may cross into Divisions B, C, or D.

**Initial state:** 25 burning cells in Division A (rows 2–6, cols 2–6)

**Strategy:** Assign crews to suppress boundary cells and build firebreaks along the Division A border. Prioritize boundary cells over interior cells.

**Score formula:** `(cells_contained_in_A / total_burning_cells) × speed_bonus`

| Score | Meaning |
|-------|---------|
| 1.0 | Fire fully contained within Division A |
| 0.5–0.9 | Fire partially crossed borders |
| 0.0–0.4 | Fire spread into 2 or more divisions |

---

### Task 2 — Asset Protection `[MEDIUM]`

**Max steps:** 50 | **Wind:** Fixed South, speed 0.5

Fire starts on the **north edge** of the map (10 burning cells) and spreads south toward 3 named assets. The agent must suppress the fire AND keep all assets safe.

**Assets to protect:**

| Asset | Location |
|-------|----------|
| Town | (15, 10) |
| Water Tower | (12, 5) |
| Road Junction | (14, 15) |

**Strategy:** Identify which asset is most at immediate risk each step. Balance suppression vs firebreak placement around assets. Coordinate crews across multiple divisions.

**Score formula:** `containment_ratio × (assets_saved / 3)`

| Score | Meaning |
|-------|---------|
| 1.0 | Fire contained AND all 3 assets safe |
| 0.6–0.9 | Fire mostly contained, 2 of 3 assets safe |
| 0.3–0.5 | Fire partially contained, 1 asset safe |
| 0.0–0.2 | All assets burned or no meaningful containment |

---

### Task 3 — Multi-Front Outbreak `[HARD]`

**Max steps:** 60 | **Wind:** Seeded deterministic (seed=42): South, speed 0.57

Fire ignites **simultaneously in Divisions A, B, and C** (8 cells per division). The agent must coordinate all Division Supervisors and Crew Bosses to contain all three fronts while protecting 6 assets and preventing any two fronts from merging.

**Assets to protect (2 per active division):**

| Asset | Location |
|-------|----------|
| Asset A1 | (7, 7) |
| Asset A2 | (8, 2) |
| Asset B1 | (7, 17) |
| Asset B2 | (8, 12) |
| Asset C1 | (17, 7) |
| Asset C2 | (18, 2) |

**Strategy:** Triage which division front is most dangerous each step. Allocate more crew resources to highest-risk division. Prevent any single front from merging with another — each merge incurs a −0.8 penalty.

**Score formula:** `avg_containment × speed_factor × asset_factor − merge_penalty`

| Score | Meaning |
|-------|---------|
| 1.0 | All 3 fronts contained, all assets safe, within 60 steps |
| 0.7–0.9 | 2 fronts fully contained, 1 partially |
| 0.4–0.6 | Mixed results across fronts |
| 0.0–0.3 | Fire spread uncontrolled across majority of map |

---

## 🛠️ Setup & Usage

### Requirements

- Python 3.10+
- Docker (for containerized deployment)

### Installation

```bash
git clone https://huggingface.co/spaces/ayushrwt/OpenEnv_Hackathon
cd OpenEnv_Hackathon
pip install -r requirements.txt
```

### Environment Variables

```bash
export API_BASE_URL="https://api.openai.com/v1"   # LLM API base URL
export API_KEY="your-key-here"                     # or HF_TOKEN for HF Inference
export MODEL_NAME="gpt-4o-mini"                    # Model to use for inference
export EVAL_BASE_URL="http://localhost:8000"        # Env server URL
```

### Running the Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment: `{"task": 1}` |
| `/step` | POST | Take one action step |
| `/state` | GET | Get full internal state snapshot |
| `/tasks` | GET | List all available tasks |
| `/grader` | GET | Get scores for all tasks |
| `/baseline` | GET | Run baseline agent |

### Running Inference

```bash
# Run a single task
python inference.py --task 1

# Run all 3 tasks
python inference.py
```

### Running Evaluation

```bash
# Evaluate all 3 tasks, 5 episodes each (default)
python evaluate.py

# Evaluate a specific task
python evaluate.py --task 2

# Custom number of episodes
python evaluate.py --episodes 10

# Specific task with custom episodes
python evaluate.py --task 3 --episodes 5
```

### Docker

```bash
docker build -t wildfire-env .
docker run -p 8000:8000 wildfire-env
```

---

## 📊 Baseline Scores

Baseline scores produced using `gpt-4o-mini` with a greedy single-step prompt (no planning, no memory), averaged over 5 episodes.

| Task | Difficulty | Baseline Score | Success Rate | Notes |
|------|------------|---------------|--------------|-------|
| Task 1 — Single Front Containment | Easy | 0.62 | 60% | Correctly suppresses boundary but misses firebreaks |
| Task 2 — Asset Protection | Medium | 0.41 | 40% | Saves 1–2 assets on average, struggles with multi-division coordination |
| Task 3 — Multi-Front Outbreak | Hard | 0.23 | 20% | Frequently allows front merges, poor resource allocation |

A score of **1.0** on all tasks is achievable with a planning-capable agent that reasons about wind direction, asset proximity, and crew positioning across multiple divisions simultaneously.

---

## 📁 File Structure

```
├── server/
│   ├── __init__.py
│   ├── app.py                              # FastAPI server — all endpoints
│   ├── agent.py                            # Agent helper logic
│   ├── tasks.py                            # Task definitions + TaskGrader
│   └── wildfireEnvironment_environment.py  # Core env: reset(), step(), state()
├── static/                                 # Static assets for UI
├── app.py                                  # Gradio UI entry point
├── gradio_ui.py                            # Gradio UI components
├── client.py                               # WildfireClient HTTP wrapper
├── inference.py                            # LLM inference script
├── evaluate.py                             # Evaluation script (all tasks)
├── models.py                               # Pydantic models: Observation, Action, Reward
├── openenv.yaml                            # OpenEnv metadata & task registry
├── Dockerfile                              # Container setup
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

---

## 🏷️ Tags

`wildfire` · `ICS` · `multi-agent` · `hierarchical-rl` · `openenv` · `emergency-response` · `Meta Hackathon 2026`
