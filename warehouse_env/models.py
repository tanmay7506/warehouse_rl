"""
Typed Pydantic models for the Warehouse RL Environment.
Defines Action, Observation, and State contracts per OpenEnv spec.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Tuple
from pydantic import BaseModel, Field
from openenv.core.env_server.types import State as BaseState


# ── Action ────────────────────────────────────────────────────────────────────

class WarehouseAction(BaseModel):
    """
    A single natural-language command string the LLM agent emits.
    Valid values: move('N'|'S'|'E'|'W'), pick(), place()
    """
    command: str = Field(
        ...,
        description="LLM-emitted command string, e.g. \"move('N')\", \"pick()\", \"place()\"",
        examples=["move('N')", "pick()", "place()"],
    )


# ── Observation ───────────────────────────────────────────────────────────────

class AgentInfo(BaseModel):
    id: int
    x: int
    y: int
    carrying_item_id: Optional[int] = None
    target_x: int
    target_y: int


class WarehouseObservation(BaseModel):
    """
    Rich observation returned after every reset() / step().
    Contains both a visual RGB frame (base64 PNG) and structured state
    so the agent can reason from text or image.
    """
    # Visual frame — base64-encoded PNG (256×256×3 RGB)
    rgb_frame: str = Field(
        ...,
        description="Base64-encoded PNG image of the current board state (256x256 RGB)"
    )
    # Structured state for text-based reasoning
    step: int = Field(0, description="Current step number within the episode")
    task_level: int = Field(1, description="Task difficulty: 1=Easy, 2=Medium, 3=Hard")
    agents: List[AgentInfo] = Field(default_factory=list)
    items_on_grid: int = Field(0, description="Number of items still undelivered on grid")
    items_delivered: int = Field(0, description="Items successfully delivered this episode")
    items_to_deliver: int = Field(0, description="Total items needed for task completion")
    dropoff_locations: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="item_id -> [dropoff_x, dropoff_y]"
    )
    # Reward and status
    reward: float = Field(0.0, description="Reward received in this step")
    done: bool = Field(False, description="True when episode is complete")
    success: bool = Field(False, description="True when task objective was completed")
    message: str = Field("", description="Human-readable status message")
    # Grader score (0.0–1.0) set by graders
    score: float = Field(0.0, description="Grader score for this step, 0.0–1.0")


# ── State ─────────────────────────────────────────────────────────────────────

class WarehouseState(BaseState):
    """
    Episode-level metadata tracked by the environment server.
    """
    task_level: int = 1
    items_delivered: int = 0
    items_to_deliver: int = 0
    total_reward: float = 0.0
    is_success: bool = False
