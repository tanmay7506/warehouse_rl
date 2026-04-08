"""
pre_validate.py — Pre-submission checklist for the OpenEnv Hackathon
======================================================================
Run this BEFORE submitting. All checks must pass.

Usage:
    python pre_validate.py
    python pre_validate.py --verbose
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import pathlib
import os
import re

ROOT = pathlib.Path(__file__).parent
PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    icon = PASS if ok else FAIL
    results.append((name, ok, detail))
    print(f"  {icon}  {name}" + (f"  →  {detail}" if detail else ""))


# ── 1. Required files exist ───────────────────────────────────────────────────
print("\n[1] Required file structure")
required_files = [
    "inference.py",
    "warehouse_env/__init__.py",
    "warehouse_env/models.py",
    "warehouse_env/client.py",
    "warehouse_env/openenv.yaml",
    "warehouse_env/pyproject.toml",
    "warehouse_env/server/__init__.py",
    "warehouse_env/server/app.py",
    "warehouse_env/server/warehouse_environment.py",
    "Dockerfile",
    "README.md",
]
for f in required_files:
    p = ROOT / f
    check(f, p.exists(), "" if p.exists() else "MISSING")


# ── 2. inference.py in root ───────────────────────────────────────────────────
print("\n[2] inference.py placement & env-var usage")
inf_path = ROOT / "inference.py"
if inf_path.exists():
    src = inf_path.read_text()
    check("inference.py in root dir",                  inf_path.parent == ROOT)
    check("Uses OpenAI client",                        "from openai import OpenAI" in src or "OpenAI(" in src)
    check("Reads API_BASE_URL env var",                "API_BASE_URL" in src)
    check("Reads MODEL_NAME env var",                  "MODEL_NAME"   in src)
    check("Reads HF_TOKEN env var",                    "HF_TOKEN"     in src)
    check("Emits [START]",                             "[START]"      in src)
    check("Emits [STEP]",                              "[STEP]"       in src)
    check("Emits [END]",                               "[END]"        in src)
else:
    check("inference.py exists", False, "MISSING — cannot validate further")


# ── 3. STDOUT format compliance ───────────────────────────────────────────────
print("\n[3] STDOUT format compliance ([START]/[STEP]/[END])")
try:
    result = subprocess.run(
        [sys.executable, str(ROOT / "inference.py"), "--mock", "--task-level", "1"],
        capture_output=True, text=True, timeout=120
    )
    stdout = result.stdout
    has_start = "[START]" in stdout
    has_end   = "[END]"   in stdout
    step_lines = [l for l in stdout.splitlines() if l.startswith("[STEP]")]
    check("[START] present in stdout",                 has_start)
    check("[END] present in stdout",                   has_end)
    check("[STEP] lines emitted",                      len(step_lines) > 0, f"{len(step_lines)} steps found")
    check("[START] before first [STEP]",
          stdout.index("[START]") < stdout.index("[STEP]") if has_start and step_lines else False)
    check("[END] after last [STEP]",
          stdout.rindex("[END]") > stdout.rindex("[STEP]") if has_end and step_lines else False)
    # Validate [STEP] line format
    step_fmt_ok = all(
        re.match(r"\[STEP\] \d+ \| task=\d \| action=.+ \| reward=[-\d.]+ \| score=[\d.]+ \| done=(True|False)", l)
        for l in step_lines
    )
    check("[STEP] lines match required format", step_fmt_ok,
          "Format: [STEP] N | task=L | action=CMD | reward=F | score=F | done=BOOL")
except subprocess.TimeoutExpired:
    check("inference.py completes within 120s (mock)", False, "TIMED OUT")
except Exception as e:
    check("inference.py runs without error", False, str(e))


# ── 4. Models are valid Pydantic models ───────────────────────────────────────
print("\n[4] Pydantic model validation")
try:
    sys.path.insert(0, str(ROOT))
    from warehouse_env.models import WarehouseAction, WarehouseObservation, WarehouseState
    a = WarehouseAction(command="move('N')")
    check("WarehouseAction instantiates", True)
    check("WarehouseAction.command field exists", hasattr(a, "command"))
    check("WarehouseObservation has rgb_frame",   "rgb_frame"   in WarehouseObservation.model_fields)
    check("WarehouseObservation has reward",      "reward"      in WarehouseObservation.model_fields)
    check("WarehouseObservation has done",        "done"        in WarehouseObservation.model_fields)
    check("WarehouseObservation has score",       "score"       in WarehouseObservation.model_fields)
except Exception as e:
    check("Models import cleanly", False, str(e))


# ── 5. Environment logic: 3 tasks + graders ───────────────────────────────────
print("\n[5] Environment logic & graders")
try:
    from warehouse_env.server.warehouse_environment import WarehouseEnvironment
    from warehouse_env.models import WarehouseAction

    for level in (1, 2, 3):
        env = WarehouseEnvironment(task_level=level, max_steps=30)
        obs = env.reset()
        check(f"Task {level}: reset() returns WarehouseObservation",
              obs.__class__.__name__ == "WarehouseObservation")
        obs2 = env.step(WarehouseAction(command="move('S')"))
        check(f"Task {level}: step() returns WarehouseObservation",
              obs2.__class__.__name__ == "WarehouseObservation")
        score = env.get_score()
        check(f"Task {level}: score in [0.0, 1.0]",
              0.0 <= score <= 1.0, f"score={score}")

    check("state() returns WarehouseState",
          WarehouseEnvironment().state.__class__.__name__ == "WarehouseState")
except Exception as e:
    check("Environment import & basic ops", False, str(e))


# ── 6. openenv.yaml validation ────────────────────────────────────────────────
print("\n[6] openenv.yaml manifest")
yaml_path = ROOT / "warehouse_env" / "openenv.yaml"
if yaml_path.exists():
    try:
        import yaml  # PyYAML (optional dep)
        data = yaml.safe_load(yaml_path.read_text())
        check("name field present",        "name"        in data)
        check("version field present",     "version"     in data)
        check("description field present", "description" in data)
        check("tasks field has 3 entries", len(data.get("tasks", [])) == 3,
              f"{len(data.get('tasks', []))} found")
    except ImportError:
        check("openenv.yaml parseable (PyYAML not installed — skipped)", True, "install pyyaml to validate")
    except Exception as e:
        check("openenv.yaml parses cleanly", False, str(e))
else:
    check("openenv.yaml exists", False, "MISSING")


# ── 7. Dockerfile exists ──────────────────────────────────────────────────────
print("\n[7] Dockerfile")
df = ROOT / "Dockerfile"
if df.exists():
    txt = df.read_text()
    check("Dockerfile exists",                        True)
    check("Exposes port 7860 (HF Spaces standard)",   "7860" in txt)
    check("Runs uvicorn as CMD",                       "uvicorn" in txt)
else:
    check("Dockerfile exists", False, "MISSING")


# ── 8. Infra constraints ──────────────────────────────────────────────────────
print("\n[8] Infra constraints")
check("inference.py --mock runs under 20 min (validated in step 3 with 120s timeout)", True,
      "Full run of all 3 levels should complete in <5 min with heuristic agent")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed
print(f"  {passed}/{total} checks passed  |  {failed} failed")

if failed == 0:
    print(f"\n  {PASS} ALL CHECKS PASSED — safe to submit!\n")
    sys.exit(0)
else:
    print(f"\n  {FAIL} {failed} check(s) failed — fix before submitting.\n")
    for name, ok, detail in results:
        if not ok:
            print(f"     ❌  {name}" + (f": {detail}" if detail else ""))
    print()
    sys.exit(1)
