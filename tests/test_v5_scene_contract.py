from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("AUTH_REQUIRED", "false")
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11435")

from app.services.document_understanding import extract_instruction_blocks, is_metadata_page
from app.services.production_planner import build_tutorial_plan
from app.services.qa_service import validate_plan
from app.services.targeting import ground_action, target_queries


class FakeText:
    class Spec:
        model = "fake-text"

    spec = Spec()

    def generate_json(self, prompt):
        if "Extract only actionable" in prompt:
            return {
                "items": [
                    {
                        "step_number": "2",
                        "action": "Open Settings",
                        "context": "Open the settings panel before continuing.",
                    }
                ]
            }
        return {"narrations": []}


class EmptyVision:
    class Spec:
        model = "fake-vision"

    spec = Spec()

    def generate_json(self, prompt, image_b64):
        return {"candidates": []}


class V5SceneContractTests(unittest.TestCase):
    def test_metadata_page_produces_no_instruction_blocks(self):
        page = {
            "page": 1,
            "heading": "Table of Contents",
            "text": "Contents\n1 Overview ........ 2\n2 Create Item ........ 4",
        }
        self.assertTrue(is_metadata_page(page))
        self.assertEqual(extract_instruction_blocks(page), [])

    def test_explanation_text_is_not_an_action(self):
        page = {
            "page": 2,
            "heading": "Item",
            "text": (
                "An Item represents a product record in the system and stores "
                "information needed throughout the workflow."
            ),
        }
        blocks = extract_instruction_blocks(page)
        self.assertFalse(any(b.get("action") for b in blocks))

    def test_numbered_action_is_still_an_action(self):
        page = {
            "page": 3,
            "heading": "Create Item",
            "text": "1. Click Create\n2. Select Part",
        }
        blocks = extract_instruction_blocks(page)
        actions = [b["action"] for b in blocks if b.get("action")]
        self.assertEqual(actions, ["1. Click Create", "2. Select Part"])

    def test_plan_emits_slide_only_explanation(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            data = {
                "job_dir": str(d),
                "title": "Item Tutorial",
                "pages": [
                    {
                        "page": 1,
                        "heading": "Item",
                        "width": 1000,
                        "height": 700,
                        "text": (
                            "An Item represents a product record in the system and "
                            "stores information needed throughout the workflow."
                        ),
                    }
                ],
                "screenshots": [],
            }
            with patch(
                "app.services.production_planner.create_clients",
                return_value=(FakeText(), EmptyVision()),
            ):
                plan = build_tutorial_plan(data)

        self.assertEqual(plan["schema_version"], 5)
        explanations = [s for s in plan["steps"] if s["kind"] == "explanation"]
        self.assertTrue(explanations)
        for scene in explanations:
            self.assertEqual(scene["visual_role"], "slide")
            self.assertEqual(scene["screenshot"], None)
            self.assertFalse(scene["cursor_enabled"])
            self.assertEqual(scene["interaction"], "none")
            self.assertEqual(scene["target_status"], "not_applicable")

    def test_plan_has_explicit_transition_scene_with_no_cursor(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            data = {
                "job_dir": str(d),
                "title": "Item Tutorial",
                "pages": [
                    {
                        "page": 1,
                        "heading": "1. Item",
                        "width": 1000,
                        "height": 700,
                        "text": "An Item represents a product record in the system and stores workflow information.",
                    }
                ],
                "screenshots": [],
            }
            with patch(
                "app.services.production_planner.create_clients",
                return_value=(FakeText(), EmptyVision()),
            ):
                plan = build_tutorial_plan(data)

        transitions = [s for s in plan["steps"] if s["kind"] == "transition"]
        self.assertTrue(transitions)
        for scene in transitions:
            self.assertEqual(scene["visual_role"], "slide")
            self.assertFalse(scene["cursor_enabled"])
            self.assertEqual(scene["screenshot"], None)

    def test_two_actions_keep_independent_source_step_keys_and_sequence(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            shot = d / "ui.png"
            Image.new("RGB", (900, 650), "white").save(shot)
            data = {
                "job_dir": str(d),
                "title": "Sequence Test",
                "pages": [
                    {
                        "page": 1,
                        "heading": "Create Item",
                        "width": 900,
                        "height": 650,
                        "text": "1. Click Create\n2. Select Part",
                        "visual_page_role": "ui_page",
                    }
                ],
                "screenshots": [
                    {
                        "page": 1,
                        "path": str(shot),
                        "asset_id": "shot-1",
                        "visual_role": "screenshot_candidate",
                        "is_full_page": False,
                        "annotation_boxes": [],
                    }
                ],
            }
            with patch(
                "app.services.production_planner.create_clients",
                return_value=(FakeText(), EmptyVision()),
            ), patch(
                "app.services.production_planner.ground_action",
                side_effect=[
                    {
                        "status": "verified",
                        "review_required": False,
                        "point": [0.20, 0.30],
                        "points": [[0.20, 0.30]],
                        "source": "ocr",
                        "confidence": 0.95,
                        "target_name": "Create",
                        "bounding_box": [0.15, 0.25, 0.25, 0.35],
                        "screenshot": {"path": str(shot), "asset_id": "shot-1", "page": 1, "is_full_page": False, "visual_role": "screenshot_candidate"},
                        "debug": {},
                        "query": "Create",
                    },
                    {
                        "status": "verified",
                        "review_required": False,
                        "point": [0.70, 0.75],
                        "points": [[0.70, 0.75]],
                        "source": "ocr",
                        "confidence": 0.95,
                        "target_name": "Part",
                        "bounding_box": [0.65, 0.70, 0.75, 0.80],
                        "screenshot": {"path": str(shot), "asset_id": "shot-1", "page": 1, "is_full_page": False, "visual_role": "screenshot_candidate"},
                        "debug": {},
                        "query": "Part",
                    },
                ],
            ):
                plan = build_tutorial_plan(data)

        actions = [s for s in plan["steps"] if s["kind"] == "action"]
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0]["source_step_key"], "1")
        self.assertEqual(actions[1]["source_step_key"], "2")
        self.assertEqual([s["sequence_index"] for s in actions], [1, 2])
        self.assertEqual(actions[0]["evidence"]["asset_id"], "shot-1")
        self.assertEqual(actions[1]["evidence"]["asset_id"], "shot-1")
        self.assertNotEqual(actions[0]["cursor"], actions[1]["cursor"])

    def test_document_full_page_can_never_be_a_cursor_source(self):
        plan = {
            "schema_version": 5,
            "steps": [
                {
                    "scene_id": "scene_1",
                    "kind": "action",
                    "source_page": 1,
                    "narration": "Click Create.",
                    "interaction": "click",
                    "target_status": "verified",
                    "target_review_required": False,
                    "cursor_enabled": True,
                    "cursor": [0.5, 0.5],
                    "screenshot": "page.png",
                    "is_full_page": True,
                    "visual_page_role": "document_page",
                }
            ],
        }
        qa = validate_plan(plan)
        self.assertFalse(qa["publishable"])
        self.assertTrue(any(x["code"] == "CURSOR_ON_DOCUMENT_PAGE" for x in qa["errors"]))

    def test_keyboard_action_has_no_mouse_target_query(self):
        self.assertEqual(target_queries("Press Enter"), [])

    def test_ambiguous_target_remains_unresolved(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            shot = d / "ui.png"
            Image.new("RGB", (800, 600), "white").save(shot)
            page = {"page": 1, "width": 800, "height": 600, "visual_page_role": "ui_page"}
            screenshot = {
                "page": 1,
                "path": str(shot),
                "visual_role": "screenshot_candidate",
                "annotation_boxes": [],
            }
            with patch(
                "app.services.targeting._ocr_candidates",
                return_value=[
                    {"point": [0.2, 0.2], "bounding_box": [0.1, 0.1, 0.3, 0.3], "target_name": "Create", "confidence": 0.93},
                    {"point": [0.7, 0.7], "bounding_box": [0.6, 0.6, 0.8, 0.8], "target_name": "Create", "confidence": 0.93},
                ],
            ):
                result = ground_action("Click Create", page, [screenshot], None)
        self.assertEqual(result["status"], "unresolved")
        self.assertTrue(result["review_required"])


if __name__ == "__main__":
    unittest.main()
