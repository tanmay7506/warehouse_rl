# Warehouse Robot RL Environment

**OpenEnv Hackathon — Meta × Scaler SST × Hugging Face**

A 3-level Warehouse Robot reinforcement-learning environment where AI agents learn to navigate, pick items from 3D shelves, and deliver them to target dropoff zones.

---

## Environment Description

An 8×8 grid warehouse with 3 shelf levels (z-axis). One or more agents receive visual (RGB image) and structured observations, then output natural-language commands. The environment supports LLM-based vision-language agents as well as text-only agents.

| Task Level | Agents | Objective | Difficulty |
|---|---|---|---|
| 1 – Easy | 1 | Navigate to a target cell | Easy |
| 2 – Medium | 1 | Pick one item and deliver it | Medium |
| 3 – Hard | 3 | Cooperatively deliver all items | Hard |

---

## Action Space

String commands parsed from LLM output:

| Command | Effect |
|---|---|
| `move('N')` | Move agent one cell North |
| `move('S')` | Move agent one cell South |
| `move('E')` | Move agent one cell East |
| `move('W')` | Move agent one cell West |
| `pick()` | Pick up item at agent's current cell |
| `place()` | Place item (scores if at correct dropoff zone) |

## Observation Space

`WarehouseObservation` Pydantic model:

| Field | Type | Description |
|---|---|---|
| `rgb_frame` | `str` (base64 PNG) | 256×256 RGB visual of the grid |
| `step` | `int` | Current step number |
| `agents` | `List[AgentInfo]` | Position, carrying status, target |
| `items_on_grid` | `int` | Remaining undelivered items |
| `items_delivered` | `int` | Items successfully delivered |
| `dropoff_locations` | `dict` | item_id → [x, y] |
| `reward` | `float` | Step reward |
| `done` | `bool` | Episode completion flag |
| `score` | `float` | Grader score [0.0–1.0] |

## Reward Function

| Event | Reward |
|---|---|
| Move closer to target | +1.0 |
| Move further from target | −5.0 |
| Step cost | −0.01 |
| Collision with another agent | −10.0 |
| Successful delivery | +50.0 |
| Reach navigation target (Task 1) | +50.0 |

---

## Setup & Installation

```bash
# 1. Install dependencies
pip install openenv-core fastapi "uvicorn[standard]" pydantic numpy pillow openai

# 2. Install in editable mode
cd warehouse_env
pip install -e .
```

## Running Locally (without Docker)

```bash
# Start the server
uvicorn warehouse_env.server.app:app --host 0.0.0.0 --port 7860

# In another terminal, run inference (mock heuristic — no API key needed)
python inference.py --mock

# Run with a real LLM
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
python inference.py
```

## Running with Docker

```bash
docker build -t warehouse-env:latest .
docker run -p 7860:7860 warehouse-env:latest
```

## Running the Pre-Submission Validator

```bash
python pre_validate.py
```

All checks must pass before submitting.

## Inference STDOUT Format

```
[START]
[STEP] 0 | task=2 | action=move('E') | reward=-1.00 | score=0.0000 | done=False
[STEP] 1 | task=2 | action=pick() | reward=-1.00 | score=0.5000 | done=False
...
[END]
```

---

## Project Structure

```
.
├── inference.py              ← Hackathon inference script (root, required)
├── pre_validate.py           ← Pre-submission validation script
├── Dockerfile                ← HF Spaces deployment
├── README.md
└── warehouse_env/
    ├── __init__.py
    ├── models.py             ← Pydantic Action / Observation / State
    ├── client.py             ← OpenEnv EnvClient subclass
    ├── openenv.yaml          ← Environment manifest
    ├── pyproject.toml
    └── server/
        ├── __init__.py
        ├── app.py            ← FastAPI entry point
        └── warehouse_environment.py  ← Core environment logic + graders
```
