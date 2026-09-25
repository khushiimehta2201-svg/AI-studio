from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.qa_service import validate_rendered_artifacts, validate_plan
from app.services.video_service import render_all_sections


def make_wav(path: Path, seconds: float = 0.30) -> None:
    rate = 24000
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.zeros(frames, dtype=np.int16).tobytes())


class V5RenderIntegrationTests(unittest.TestCase):
    def test_slide_then_two_actions_keep_one_canonical_timeline(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            shot = d / "ui.png"
            audio = d / "audio.wav"
            Image.new("RGB", (900, 650), "white").save(shot)
            make_wav(audio)

            plan = {
                "schema_version": 5,
                "title": "Create Item",
                "sections": [],
                "steps": [
                    {
                        "id": 1,
                        "scene_id": "scene_00001",
                        "kind": "transition",
                        "title": "1. Item",
                        "source_page": 1,
                        "visual_role": "slide",
                        "visual_body": "Item",
                        "screenshot": None,
                        "interaction": "none",
                        "target_status": "not_applicable",
                        "cursor_enabled": False,
                        "narration": "Item.",
                        "caption_text": "Item.",
                    },
                    {
                        "id": 2,
                        "scene_id": "scene_00002",
                        "kind": "explanation",
                        "title": "What an Item represents",
                        "source_page": 1,
                        "visual_role": "slide",
                        "visual_body": "An Item represents a product record.",
                        "screenshot": None,
                        "interaction": "none",
                        "target_status": "not_applicable",
                        "cursor_enabled": False,
                        "narration": "An Item represents a product record.",
                        "caption_text": "An Item represents a product record.",
                    },
                    {
                        "id": 3,
                        "scene_id": "scene_00003",
                        "kind": "action",
                        "title": "Click Create",
                        "source_page": 1,
                        "source_step_number": "1",
                        "source_step_key": "1",
                        "sequence_index": 1,
                        "page_width": 900,
                        "page_height": 650,
                        "visual_role": "screenshot",
                        "screenshot": str(shot),
                        "interaction": "click",
                        "target_status": "verified",
                        "target_review_required": False,
                        "cursor_enabled": True,
                        "cursor": [0.70, 0.75],
                        "cursor_path": [[0.70, 0.75]],
                        "cursor_bounding_box": [0.64, 0.69, 0.76, 0.81],
                        "narration": "Click Create now.",
                        "caption_text": "Click Create now.",
                    },
                    {
                        "id": 4,
                        "scene_id": "scene_00004",
                        "kind": "action",
                        "title": "Select Part",
                        "source_page": 1,
                        "source_step_number": "2",
                        "source_step_key": "2",
                        "sequence_index": 2,
                        "page_width": 900,
                        "page_height": 650,
                        "visual_role": "screenshot",
                        "screenshot": str(shot),
                        "interaction": "click",
                        "target_status": "verified",
                        "target_review_required": False,
                        "cursor_enabled": True,
                        "cursor": [0.30, 0.35],
                        "cursor_path": [[0.30, 0.35]],
                        "cursor_bounding_box": [0.24, 0.29, 0.36, 0.41],
                        "narration": "Select Part now.",
                        "caption_text": "Select Part now.",
                    },
                ],
            }

            pre = validate_plan(plan)
            self.assertTrue(pre["publishable"], pre)

            audio_data = [
                {"index": i, "scene_id": scene["scene_id"], "path": str(audio), "duration": 0.30}
                for i, scene in enumerate(plan["steps"])
            ]
            rendered = render_all_sections(plan, audio_data, str(d))
            (d / "plan.json").write_text(json.dumps(rendered), encoding="utf-8")

            self.assertTrue((d / "tutorial.mp4").exists())
            self.assertTrue((d / "tutorial.vtt").exists())
            timeline = rendered["dialogue_timeline"]
            self.assertEqual(
                [x["scene_id"] for x in timeline],
                [x["scene_id"] for x in plan["steps"]],
            )
            self.assertEqual(timeline[0]["interaction"], "none")
            self.assertEqual(timeline[1]["interaction"], "none")
            self.assertEqual(timeline[2]["interaction"], "click")
            self.assertEqual(timeline[3]["interaction"], "click")
            self.assertNotEqual(timeline[2]["scene_id"], timeline[3]["scene_id"])

            post = validate_rendered_artifacts(str(d), rendered)
            self.assertTrue(post["publishable"], post)


if __name__ == "__main__":
    unittest.main()
