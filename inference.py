"""
inference.py — Warehouse Robot LLM Inference Script
====================================================
Hackathon-compliant inference loop for the OpenEnv Warehouse environment.

Required environment variables (set before running):
    API_BASE_URL  - LLM API endpoint  (e.g. https://api-inference.huggingface.co/v1)
    MODEL_NAME    - Model identifier  (e.g. meta-llama/Llama-3.1-8B-Instruct)
    HF_TOKEN      - Hugging Face / API key

STDOUT contract (evaluated programmatically — do NOT change format):
    [START]
    [STEP] <n> | task=<level> | action=<cmd> | reward=<float> | score=<float> | done=<bool>
    [END]

Usage:
    python inference.py                          # runs all 3 task levels
    python inference.py --task-level 2           # single level
    python inference.py --mock                   # heuristic agent (no LLM needed)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit(
        "openai package not installed. Run: pip install openai"
    )

try:
    from PIL import Image
except ImportError:
    raise SystemExit("pillow not installed. Run: pip install pillow")

# ── Local env (direct import — works when running from project root) ───────────
# We import the core logic directly to avoid needing a running server for
# the baseline inference script (saves infra for the 2-vCPU / 8 GB constraint).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from warehouse_env.server.warehouse_environment import WarehouseEnvironment
from warehouse_env.models import WarehouseAction

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN:     str = os.environ.get("HF_TOKEN",     "")

MAX_STEPS_PER_EPISODE = 50   # keep total runtime well under 20 min
FALLBACK_ACTION       = "move('S')"

SYSTEM_PROMPT = """\
You are controlling a Warehouse Robot on an 8×8 grid.
You receive a description of the current state.

Your only valid actions are:
  move('N')  move('S')  move('E')  move('W')
  pick()
  place()

Rules:
- move() shifts the robot one cell in that cardinal direction.
- pick() picks up an item if the robot is standing on one.
- place() drops the item at the current cell; only scores if at the correct dropoff zone.
- You CANNOT walk off the grid or through other agents.

Respond with EXACTLY one action and nothing else.
Examples of valid responses:  move('N')   pick()   place()
"""


# ── OpenAI client setup ───────────────────────────────────────────────────────

def _build_client() -> Optional[OpenAI]:
    """Return an OpenAI-compatible client, or None if no key is available."""
    if not HF_TOKEN:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ── Image helpers ─────────────────────────────────────────────────────────────

def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ── LLM action selection ──────────────────────────────────────────────────────

def _parse_action(text: str) -> str:
    """Extract first valid action command from LLM text output."""
    m = re.search(
        r"(move\s*\(\s*['\"][NSEW]['\"]?\s*\)|pick\s*\(\s*\)|place\s*\(\s*\))",
        text,
        re.IGNORECASE,
    )
    return m.group(0) if m else FALLBACK_ACTION


def _llm_action(
    client: OpenAI,
    obs,          # WarehouseObservation
) -> str:
    """Call the LLM with a text-only state description and return the action."""
    # Build a compact text description (avoids sending large base64 images in
    # every call — keeps latency low on small models).
    agent = obs.agents[0] if obs.agents else None
    state_text = (
        f"Step {obs.step}/{MAX_STEPS_PER_EPISODE}. "
        f"Task level {obs.task_level}. "
        f"Agent at ({agent.x},{agent.y}), "
        f"carrying={'item '+str(agent.carrying_item_id) if agent and agent.carrying_item_id is not None else 'nothing'}. "
        f"Target cell: ({agent.target_x},{agent.target_y}). "
        f"Items on grid: {obs.items_on_grid}. "
        f"Items delivered: {obs.items_delivered}/{obs.items_to_deliver}. "
        f"Dropoff zones: {obs.dropoff_locations}. "
        "What is the best next action?"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": state_text},
            ],
            max_tokens=20,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        return _parse_action(raw)
    except Exception as exc:
        # Fallback — do not crash the eval loop
        print(f"[WARN] LLM call failed: {exc}", file=sys.stderr, flush=True)
        return FALLBACK_ACTION


def _heuristic_action(obs) -> str:
    """
    Simple rule-based heuristic used when --mock flag is set or no API key.
    Useful to prove STDOUT compliance without any LLM.
    """
    if not obs.agents:
        return FALLBACK_ACTION
    ag = obs.agents[0]

    # If carrying an item, move toward dropoff
    if ag.carrying_item_id is not None:
        dropoff = obs.dropoff_locations.get(str(ag.carrying_item_id))
        if dropoff:
            if ag.x < dropoff[0]: return "move('E')"
            if ag.x > dropoff[0]: return "move('W')"
            if ag.y < dropoff[1]: return "move('S')"
            if ag.y > dropoff[1]: return "move('N')"
            return "place()"  # on top of dropoff
    else:
        # Move toward target (item or nav target)
        tx, ty = ag.target_x, ag.target_y
        if ag.x == tx and ag.y == ty:
            return "pick()"   # try to pick (task 1: triggers nav completion)
        if ag.x < tx: return "move('E')"
        if ag.x > tx: return "move('W')"
        if ag.y < ty: return "move('S')"
        if ag.y > ty: return "move('N')"

    return FALLBACK_ACTION


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_level: int, client: Optional[OpenAI], mock: bool) -> dict:
    """
    Run a single episode.
    Returns a summary dict with keys: task_level, steps, score, success.
    """
    env = WarehouseEnvironment(task_level=task_level, max_steps=MAX_STEPS_PER_EPISODE)
    obs = env.reset()

    total_reward = 0.0
    steps_taken  = 0

    for step in range(MAX_STEPS_PER_EPISODE):
        # Choose action
        if mock or client is None:
            action_str = _heuristic_action(obs)
        else:
            action_str = _llm_action(client, obs)

        # Step the environment
        obs = env.step(WarehouseAction(command=action_str))

        total_reward += obs.reward
        steps_taken   = step + 1

        # ── REQUIRED STDOUT FORMAT ────────────────────────────────────────────
        print(
            f"[STEP] {step} | task={task_level} | action={action_str} "
            f"| reward={obs.reward:.2f} | score={obs.score:.4f} | done={obs.done}",
            flush=True,
        )
        # ─────────────────────────────────────────────────────────────────────

        if obs.done:
            break

    final_score = env.get_score()
    return {
        "task_level":   task_level,
        "steps":        steps_taken,
        "total_reward": total_reward,
        "score":        final_score,
        "success":      obs.success,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Warehouse Env inference runner")
    parser.add_argument(
        "--task-level", type=int, default=0,
        help="1, 2, or 3 to run a single level; 0 (default) runs all three",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use heuristic agent instead of LLM (no API key needed)",
    )
    args = parser.parse_args()

    client = None if args.mock else _build_client()
    if client is None and not args.mock:
        print(
            "[WARN] HF_TOKEN not set — falling back to heuristic agent.",
            file=sys.stderr, flush=True,
        )

    task_levels = [args.task_level] if args.task_level in (1, 2, 3) else [1, 2, 3]

    # ── REQUIRED: [START] before any steps ────────────────────────────────────
    print("[START]", flush=True)

    all_results = []
    for level in task_levels:
        result = run_episode(task_level=level, client=client, mock=args.mock)
        all_results.append(result)

    # ── REQUIRED: [END] after all steps ───────────────────────────────────────
    print("[END]", flush=True)

    # Human-readable summary (goes to stderr so it doesn't pollute grader stdout)
    print("\n── Inference Summary ──", file=sys.stderr)
    for r in all_results:
        print(
            f"  Task {r['task_level']}: steps={r['steps']} "
            f"score={r['score']:.4f} success={r['success']} "
            f"total_reward={r['total_reward']:.1f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
