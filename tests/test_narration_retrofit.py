import importlib
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


class NarrationRetrofitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tests are intended to run from the repository root.
        cls.ai = importlib.import_module("app.services.ai_service")
        cls.tts = importlib.import_module("app.services.tts_service")
        cls.video = importlib.import_module("app.services.video_service")

    def setUp(self):
        self.ai._NARRATION_CACHE.clear()
        self.ai._OLLAMA_NARRATION_DISABLED = False

    def test_heading_that_is_action_is_not_deleted(self):
        page = {
            "heading": "1. Click the File menu",
            "text": (
                "1. Click the File menu\n"
                "This opens the document options."
            ),
        }

        blocks = self.ai._extract_instruction_blocks(page)
        actions = [
            b["action"] for b in blocks if b.get("action")
        ]

        self.assertEqual(
            actions,
            ["1. Click the File menu"],
        )
        self.assertEqual(
            blocks[0]["context"],
            "This opens the document options.",
        )

    def test_wrapped_numbered_step_is_one_action(self):
        page = {
            "text": (
                "1. Click the Configuration\n"
                "menu to continue.\n"
                "2. Select Settings."
            ),
        }

        actions, _ = self.ai._extract_action_lines(page)

        self.assertEqual(
            actions,
            [
                "1. Click the Configuration menu to continue.",
                "2. Select Settings.",
            ],
        )

    def test_explanatory_sentence_does_not_become_action(self):
        page = {
            "text": (
                "The configuration panel controls system preferences.\n"
                "The download starts after validation."
            ),
        }

        actions, explanations = self.ai._extract_action_lines(page)

        self.assertEqual(actions, [])
        self.assertTrue(explanations)

    def test_local_ollama_narration_is_contextual_and_url_free(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "response": (
                        '{"narration":"Click the documentation link to '
                        'review the configuration details."}'
                    )
                }

        def fake_post(*args, **kwargs):
            self.assertNotIn(
                "https://private.example/docs",
                kwargs["json"]["prompt"],
            )
            self.assertEqual(
                kwargs["json"]["model"],
                self.ai.OLLAMA_MODEL,
            )
            return FakeResponse()

        with patch.object(self.ai.requests, "post", fake_post):
            result = self.ai._make_action_narration(
                "Click the documentation link: https://private.example/docs",
                "Use this page to review the configuration details.",
            )

        self.assertEqual(
            result,
            "Click the documentation link to review the configuration details.",
        )
        self.assertNotIn("http://", result)
        self.assertNotIn("https://", result)

    def test_plan_keeps_urls_in_caption_but_not_tts(self):
        self.ai._OLLAMA_NARRATION_DISABLED = True

        data = {
            "pages": [
                {
                    "page": 1,
                    "heading": "Configuration",
                    "text": (
                        "Configuration\n"
                        "This panel controls system preferences.\n"
                        "1. Click the Settings button.\n"
                        "This opens the settings form.\n"
                        "URL: https://example.com/docs"
                    ),
                    "width": 100,
                    "height": 100,
                    "words": [],
                    "urls": ["https://example.com/docs"],
                }
            ],
            "screenshots": [],
        }

        plan = self.ai.build_tutorial_plan(data)
        step = plan["steps"][0]

        self.assertIn("settings form", step["narration"].lower())
        self.assertNotRegex(
            step["tts_narration"],
            r"https?://|www\.",
        )
        self.assertIn(
            "https://example.com/docs",
            step["caption_urls"],
        )
        self.assertEqual(step["caption"], step["caption_text"])
        self.assertEqual(
            step["urls"],
            ["https://example.com/docs"],
        )

    def test_tts_defensively_removes_urls(self):
        self.assertEqual(
            self.tts._remove_urls(
                "Open https://example.com/docs and continue."
            ),
            "Open and continue.",
        )

    def test_concat_audio_matches_visual_scene_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            audio_paths = []

            for index, seconds in enumerate((1.0, 4.0, None, 2.0)):
                if seconds is None:
                    audio_paths.append(None)
                    continue

                path = output_dir / f"input_{index}.wav"
                with wave.open(str(path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(
                        b"\x00"
                        * int(round(seconds * 24000))
                        * 2
                    )
                audio_paths.append(str(path))

            scene_timings = [
                {
                    "duration": self.video._scene_duration(path)[0],
                    "narration_start": 0.55,
                }
                for path in audio_paths
            ]

            output = self.video._concat_audio(
                audio_paths,
                scene_timings,
                output_dir,
                "test",
            )
            scene_durations = [item["duration"] for item in scene_timings]

            self.assertIsNotNone(output)

            with wave.open(str(output), "rb") as wf:
                actual = (
                    wf.getnframes()
                    / float(wf.getframerate())
                )

            expected = sum(scene_durations)
            self.assertAlmostEqual(
                actual,
                expected,
                places=6,
            )

    def test_concat_audio_accepts_legacy_float_scene_durations(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            output = self.video._concat_audio(
                [None, None],
                [2.0, 3.0],
                output_dir,
                "legacy",
            )
            self.assertIsNotNone(output)
            with wave.open(str(output), "rb") as wf:
                actual = wf.getnframes() / float(wf.getframerate())
            self.assertAlmostEqual(actual, 5.0, places=6)

    def test_uncertain_target_disables_cursor_and_click(self):
        plan = self.ai.build_tutorial_plan({
            "pages": [{
                "page": 1,
                "heading": "Open Settings",
                "text": "Open Settings",
                "width": 100,
                "height": 100,
            }],
            "screenshots": [{
                "page": 1,
                "path": "missing.png",
                "is_full_page": False,
            }],
        })
        step = plan["steps"][0]
        self.assertFalse(step["cursor_enabled"])
        self.assertEqual(step["interaction"], "none")
        self.assertIsNone(step["cursor"])

    def test_click_timing_is_after_narration_starts(self):
        timing = self.video._timing_plan(
            {"interaction": "click"},
            3.0,
            (900, 400),
            None,
            False,
        )
        self.assertGreaterEqual(timing["interaction_time"], timing["narration_start"])

    def test_same_screenshot_continues_cursor_position(self):
        timing = self.video._timing_plan(
            {"interaction": "click"},
            2.0,
            (500, 300),
            (100, 100),
            True,
        )
        self.assertEqual(timing["start_x"], 100.0)
        self.assertEqual(timing["start_y"], 100.0)


    def test_keyboard_press_reports_no_mouse_target(self):
        plan = self.ai.build_tutorial_plan({
            "pages": [{
                "page": 1, "text": "Press Enter to continue.", "heading": "Press Enter to continue.",
                "width": 100, "height": 100, "words": [{"text": "Enter", "x0": 10, "y0": 10, "x1": 30, "y1": 20}],
            }], "screenshots": []
        })
        step = plan["steps"][0]
        self.assertFalse(step["cursor_enabled"])
        self.assertEqual(step["interaction"], "none")
        self.assertEqual(step["target_status"], "not_applicable")

    def test_scene_contains_canonical_interaction_plan_after_render(self):
        import tempfile
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "ui.png"
            cv2.imwrite(str(img), np.full((120, 200, 3), 240, dtype=np.uint8))
            wav = root / "step.wav"
            with wave.open(str(wav), "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000); wf.writeframes(b"\0" * 24000 * 2)
            plan = {"steps": [{
                "id": 1, "kind": "action", "title": "Click Settings", "narration": "Click Settings.",
                "caption_text": "Click Settings.", "screenshot": str(img), "page_width": 200, "page_height": 120,
                "cursor": [0.5, 0.5], "cursor_enabled": True, "target_status": "verified",
                "interaction": "click", "dialogue": {"current_action": "Click Settings", "brief": "Click Settings.", "purpose": ""}
            }]}
            self.video.render_tutorial(plan, {"files": [str(wav)]}, str(root))
            scene = plan["steps"][0]
            self.assertIn("interaction_plan", scene)
            self.assertAlmostEqual(scene["interaction_plan"]["scene_duration"], scene["timing"]["duration"], places=3)
            self.assertGreaterEqual(scene["interaction_plan"]["interaction_time"], scene["interaction_plan"]["narration_start"])

    def test_page_text_ambiguous_target_is_rejected(self):
        page = {
            "words": [
                {"text": "Settings", "x0": 10, "y0": 10, "x1": 30, "y1": 20},
                {"text": "Settings", "x0": 400, "y0": 500, "x1": 430, "y1": 510},
            ]
        }
        self.assertIsNone(self.ai._find_text_target("Settings", page))

    def test_long_caption_fits_two_lines_without_truncation(self):
        text = "Open the Administration Preferences panel to review the security configuration settings required for the current environment."
        lines, scale = self.video._caption_lines(text)
        self.assertLessEqual(len(lines), 2)
        self.assertEqual(" ".join(lines), text)
        self.assertGreaterEqual(scale, 0.40)

    def test_narration_validation_rejects_incomplete_output(self):
        self.assertFalse(self.ai._narration_is_usable("Click Settings to", "Click Settings."))
        self.assertFalse(self.ai._narration_is_usable("Click https://example.com now.", "Click Settings."))

    def test_vtt_can_include_urls_while_burned_caption_stays_clean(self):
        self.assertNotIn("https://", self.video._wrap_text("Open Settings now.")[0])


    def test_structured_pixel_target_is_normalized_before_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image
            path = Path(tmp) / 'shot.png'
            Image.new('RGB', (1000, 500), 'white').save(path)
            candidate = self.ai._candidate_target(
                'Settings',
                {
                    'path': str(path),
                    'candidates': [
                        {'text': 'Settings', 'point': [800, 250], 'box': {'x': 750, 'y': 220, 'width': 100, 'height': 60}}
                    ],
                },
            )
            self.assertTrue(candidate['accepted'])
            self.assertAlmostEqual(candidate['point'][0], 0.8)
            self.assertAlmostEqual(candidate['point'][1], 0.5)

    def test_keyboard_press_does_not_create_mouse_click(self):
        self.assertEqual(
            self.ai._interaction_for_action('Press Enter to continue.', 'Press', True),
            'none',
        )


    def test_ambiguous_candidates_are_rejected(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'shot.png'
            Image.new('RGB', (1000, 500), 'white').save(path)
            candidate = self.ai._candidate_target(
                'Settings',
                {
                    'path': str(path),
                    'candidates': [
                        {'text': 'Settings', 'point': [100, 100]},
                        {'text': 'Settings', 'point': [300, 100]},
                    ],
                },
            )
            self.assertFalse(candidate['accepted'])
            self.assertEqual(candidate['target_status'], 'ambiguous')

    def test_url_only_navigation_has_no_target_query(self):
        self.assertEqual(
            self.ai._target_query('Open https://example.com/docs'),
            '',
        )



if __name__ == "__main__":
    unittest.main()
