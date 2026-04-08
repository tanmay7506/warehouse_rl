"""
FastAPI app entry-point for the Warehouse OpenEnv environment.
Exposes WebSocket-based /ws endpoint, /health, and optional /web UI.
"""
from openenv.core.env_server import create_fastapi_app
from ..models import WarehouseAction, WarehouseObservation
from .warehouse_environment import WarehouseEnvironment

# Default to task_level=2 (medium); override via TASK_LEVEL env var if needed
import os
_task_level = int(os.environ.get("TASK_LEVEL", "2"))

env = WarehouseEnvironment(task_level=_task_level)
app = create_fastapi_app(env, WarehouseAction, WarehouseObservation)
