"""Compatibility façade for the canonical planner.

Legacy callers can continue importing build_tutorial_plan/generate_tutorial while
all generation logic lives in production_planner.py.
"""
from __future__ import annotations

from app.services.production_planner import build_tutorial_plan


generate_tutorial=build_tutorial_plan
