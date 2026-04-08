"""
Warehouse RL Environment — server-side OpenEnv implementation.
Subclasses openenv.core.env_server.Environment (not gymnasium.Env).

Task Levels
-----------
1  Easy   – 1 agent, navigate to a target cell (no item)
2  Medium – 1 agent, pick one item and deliver it to the dropoff zone
3  Hard   – 3 agents, pick & deliver N items cooperatively (collision-aware)
"""
from __future__ import annotations

import base64
import io
import random
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from openenv.core.env_server import Environment
from ..models import WarehouseAction, WarehouseObservation, WarehouseState, AgentInfo

# ── Action constants (internal) ───────────────────────────────────────────────
ACTION_N     = 0
ACTION_S     = 1
ACTION_E     = 2
ACTION_W     = 3
ACTION_PICK  = 4
ACTION_PLACE = 5

GRID_W = 8
GRID_H = 8
GRID_Z = 3   # shelf levels


@dataclass
class _AgentState:
    id: int
    x: int
    y: int
    carrying_item_id: Optional[int]
    target_pos: Tuple[int, int]


# ── Main environment ──────────────────────────────────────────────────────────

class WarehouseEnvironment(Environment):
    """
    OpenEnv-compliant warehouse robot environment.
    Accepts natural-language LLM commands via WarehouseAction.command.
    Returns WarehouseObservation on every reset() / step().
    """

    def __init__(self, task_level: int = 2, max_steps: int = 200) -> None:
        super().__init__()
        if task_level not in (1, 2, 3):
            raise ValueError("task_level must be 1, 2, or 3")
        self._task_level   = task_level
        self._max_steps    = max_steps
        self._num_agents   = 1 if task_level in (1, 2) else 3
        self._state        = WarehouseState()
        # Episode variables (set in reset)
        self._agents:             List[_AgentState]           = []
        self._grid_items:         Dict[Tuple[int,int,int],int] = {}
        self._dropoff_locations:  Dict[int, Tuple[int,int]]   = {}
        self._items_to_deliver:   int = 0
        self._items_delivered:    int = 0
        self._step_count:         int = 0
        self._task1_target:       Optional[Tuple[int,int]] = None

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(self) -> WarehouseObservation:
        self._state = WarehouseState(
            episode_id=str(uuid.uuid4()),
            task_level=self._task_level,
        )
        self._step_count     = 0
        self._agents         = []
        self._grid_items     = {}
        self._dropoff_locations = {}
        self._task1_target   = None

        spawn = [(0, 0), (0, 7), (7, 0)]

        if self._task_level == 1:
            self._items_to_deliver = 0
            self._task1_target = (
                random.randint(2, 7), random.randint(2, 7)
            )
        elif self._task_level == 2:
            self._items_to_deliver = 1
        else:
            self._items_to_deliver = self._num_agents * 2

        self._items_delivered = 0

        # Place items
        for item_id in range(self._items_to_deliver):
            x, y, z = self._rand_free_cell()
            self._grid_items[(x, y, z)] = item_id
            self._dropoff_locations[item_id] = (
                random.choice([0, 7]), random.choice([0, 7])
            )

        # Spawn agents
        for i in range(self._num_agents):
            sx, sy = spawn[i]
            if self._task_level == 1:
                tgt = self._task1_target
            else:
                tgt = self._closest_item(sx, sy) if self._grid_items else (sx, sy)
            self._agents.append(
                _AgentState(id=i, x=sx, y=sy, carrying_item_id=None, target_pos=tgt)
            )

        self._state.task_level       = self._task_level
        self._state.items_to_deliver = self._items_to_deliver

        return self._make_observation(reward=0.0, done=False)

    def step(self, action: WarehouseAction) -> WarehouseObservation:
        self._step_count += 1
        self._state.step_count = self._step_count

        agent     = self._agents[0]
        old_dist  = self._manhattan(agent.x, agent.y, *agent.target_pos)
        reward    = 0.0

        # Parse command string produced by LLM
        cmd_str = action.command.strip().lower()
        commands = re.findall(
            r"(move|pick|place)\s*\(\s*['\"]?([a-z]*)['\"]?\s*\)", cmd_str
        )

        for cmd, arg in commands:
            if cmd == "move":
                dx, dy = {"n":(0,-1),"s":(0,1),"e":(1,0),"w":(-1,0)}.get(arg, (0,0))
                nx, ny = agent.x + dx, agent.y + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    collision = any(
                        a.x == nx and a.y == ny and a.id != agent.id
                        for a in self._agents
                    )
                    if collision:
                        reward -= 10.0
                    else:
                        agent.x, agent.y = nx, ny

            elif cmd == "pick":
                if agent.carrying_item_id is None:
                    for z in range(GRID_Z):
                        if (agent.x, agent.y, z) in self._grid_items:
                            item_id = self._grid_items.pop((agent.x, agent.y, z))
                            agent.carrying_item_id = item_id
                            agent.target_pos = self._dropoff_locations[item_id]
                            break

            elif cmd == "place":
                if agent.carrying_item_id is not None:
                    item_id = agent.carrying_item_id
                    dropoff = self._dropoff_locations[item_id]
                    if agent.x == dropoff[0] and agent.y == dropoff[1]:
                        agent.carrying_item_id = None
                        self._items_delivered += 1
                        reward += 50.0
                        agent.target_pos = self._closest_item(agent.x, agent.y)
                    else:
                        # Wrong cell — put it back on nearest free shelf
                        for z in range(GRID_Z):
                            if (agent.x, agent.y, z) not in self._grid_items:
                                self._grid_items[(agent.x, agent.y, z)] = item_id
                                agent.carrying_item_id = None
                                agent.target_pos = self._closest_item(agent.x, agent.y)
                                break

        new_dist = self._manhattan(agent.x, agent.y, *agent.target_pos)
        if new_dist < old_dist:
            reward += 1.0
        elif new_dist > old_dist:
            reward -= 5.0
        reward -= 0.01  # step cost

        # Task-1 completion bonus
        if self._task_level == 1 and new_dist == 0:
            reward += 50.0
            self._items_delivered = 1

        done = self._check_done()
        success = done and (
            self._items_delivered >= max(self._items_to_deliver, 1)
        )

        self._state.items_delivered = self._items_delivered
        self._state.total_reward   += reward
        self._state.is_success      = success

        return self._make_observation(reward=reward, done=done, success=success)

    @property
    def state(self) -> WarehouseState:
        return self._state

    # ── Graders (called externally for hackathon evaluation) ──────────────────

    def grade_task1(self) -> float:
        """Easy – navigation only. Score: 1.0 if reached target, else partial."""
        if not self._agents:
            return 0.0
        agent = self._agents[0]
        if self._task1_target is None:
            return 0.0
        dist = self._manhattan(agent.x, agent.y, *self._task1_target)
        max_dist = GRID_W + GRID_H
        return max(0.0, 1.0 - dist / max_dist)

    def grade_task2(self) -> float:
        """Medium – 1 item delivery. Score: 1.0 if delivered, 0.5 if picked, else partial."""
        if self._items_delivered >= 1:
            return 1.0
        agent = self._agents[0] if self._agents else None
        if agent and agent.carrying_item_id is not None:
            return 0.5
        return 0.0

    def grade_task3(self) -> float:
        """Hard – multi-agent deliveries. Score: fraction of items delivered."""
        if self._items_to_deliver == 0:
            return 0.0
        return min(1.0, self._items_delivered / self._items_to_deliver)

    def get_score(self) -> float:
        """Return the appropriate grade for the current task level, in [0,1]."""
        return [None, self.grade_task1, self.grade_task2, self.grade_task3][self._task_level]()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_done(self) -> bool:
        if self._step_count >= self._max_steps:
            return True
        if self._task_level == 1 and self._items_delivered >= 1:
            return True
        if self._task_level > 1 and self._items_to_deliver > 0:
            return self._items_delivered >= self._items_to_deliver
        return False

    def _manhattan(self, x1, y1, x2, y2) -> int:
        return abs(x1 - x2) + abs(y1 - y2)

    def _closest_item(self, x: int, y: int) -> Tuple[int, int]:
        if not self._grid_items:
            return (x, y)
        closest = min(
            self._grid_items.keys(),
            key=lambda p: self._manhattan(x, y, p[0], p[1])
        )
        return (closest[0], closest[1])

    def _rand_free_cell(self) -> Tuple[int,int,int]:
        while True:
            x, y, z = random.randint(1,6), random.randint(1,6), random.randint(0, GRID_Z-1)
            if (x, y, z) not in self._grid_items:
                return x, y, z

    def _render_rgb_b64(self) -> str:
        """Render a 256×256 RGB board and return it as a base64 PNG string."""
        CELL = 32
        img  = np.full((256, 256, 3), 245, dtype=np.uint8)

        # Grid lines
        for i in range(1, 8):
            img[i*CELL, :] = 180
            img[:, i*CELL] = 180

        # Drop-off zones (light green)
        for drop in self._dropoff_locations.values():
            dx, dy = drop
            img[dy*CELL:(dy+1)*CELL, dx*CELL:(dx+1)*CELL] = [144, 238, 144]

        # Items on shelves (blue, z-offset)
        for (x, y, z), _ in self._grid_items.items():
            cx, cy = x*CELL, y*CELL
            off = 4 + z*6
            img[cy+off:cy+CELL-off, cx+off:cx+CELL-off] = [0, 102, 204]

        # Agents + targets
        for ag in self._agents:
            tx, ty = ag.target_pos
            img[ty*CELL+12:ty*CELL+20, tx*CELL+12:tx*CELL+20] = [255, 165, 0]  # orange target
            cx, cy = ag.x*CELL, ag.y*CELL
            img[cy+4:cy+28, cx+4:cx+28] = [220, 50, 50]  # red agent body
            if ag.carrying_item_id is not None:
                img[cy+12:cy+20, cx+12:cx+20] = [0, 255, 200]  # cyan = carrying

        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _make_observation(
        self,
        reward: float = 0.0,
        done:   bool  = False,
        success: bool = False,
    ) -> WarehouseObservation:
        agent_infos = []
        for ag in self._agents:
            agent_infos.append(AgentInfo(
                id=ag.id, x=ag.x, y=ag.y,
                carrying_item_id=ag.carrying_item_id,
                target_x=ag.target_pos[0], target_y=ag.target_pos[1],
            ))

        dropoff_serialized = {
            str(k): list(v) for k, v in self._dropoff_locations.items()
        }

        score = self.get_score() if self._agents else 0.0

        msg = "Episode running"
        if done:
            msg = "Task completed!" if success else "Episode ended (timeout or partial)"

        return WarehouseObservation(
            rgb_frame=self._render_rgb_b64(),
            step=self._step_count,
            task_level=self._task_level,
            agents=agent_infos,
            items_on_grid=len(self._grid_items),
            items_delivered=self._items_delivered,
            items_to_deliver=self._items_to_deliver,
            dropoff_locations=dropoff_serialized,
            reward=reward,
            done=done,
            success=success,
            message=msg,
            score=score,
        )
