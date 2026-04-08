"""
Client for the Warehouse OpenEnv environment.
Usage (async):
    async with WarehouseEnv(base_url="ws://localhost:8000") as env:
        obs = await env.reset()
        result = await env.step(WarehouseAction(command="move('N')"))

Usage (sync):
    with WarehouseEnv(base_url="ws://localhost:8000").sync() as env:
        obs = env.reset()
        result = env.step(WarehouseAction(command="pick()"))
"""
from openenv.core.env_client import EnvClient
from .models import WarehouseAction, WarehouseObservation

class WarehouseEnv(EnvClient[WarehouseAction, WarehouseObservation]):
    action_type      = WarehouseAction
    observation_type = WarehouseObservation
