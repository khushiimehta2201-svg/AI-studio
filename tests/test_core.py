from app.services.ai_service import (
    _candidate_screenshots,
    _choose_screenshot,
    _is_bad_segment,
    _select_screenshot_and_visual,
    _split_compound_action,
)
from app.services.video_service import (
    MIN_SCENE_SECONDS,
    SCENE_PADDING_SECONDS,
    _audio_duration_for_scene,
)


# ----------------------------------------------------------------------
# Compound action splitting
# ----------------------------------------------------------------------

def test_split_compound_action_separates_distinct_ui_interactions():
    pieces = _split_compound_action(
        "Click Add button and select ITL Design Part from drop down."
    )
    assert pieces == [
        "Click Add button.",
        "Select ITL Design Part from drop down.",
    ]


def test_split_compound_action_handles_select_then_click():
    pieces = _split_compound_action(
        "Select the part and click Attachment tab."
    )
    assert pieces == [
        "Select the part.",
        "Click Attachment tab.",
    ]


# ----------------------------------------------------------------------
# Fragment rejection
# ----------------------------------------------------------------------

def test_bare_verb_fragments_are_rejected():
    for bad in ("Type", "Select", "Click", "Open"):
        assert _is_bad_segment(bad)


def test_normal_sentence_is_not_a_bad_segment():
    assert not _is_bad_segment("Click the Filter button.")


# ----------------------------------------------------------------------
# Screenshot candidate selection
# ----------------------------------------------------------------------

SCREENSHOTS = [
    {"index": 0, "page": 1, "screenshot_index_on_page": 0, "path": "p1_a.png"},
    {"index": 1, "page": 1, "screenshot_index_on_page": 1, "path": "p1_b.png"},
    {"index": 2, "page": 2, "screenshot_index_on_page": 0, "path": "p2_a.png"},
]


def test_candidate_screenshots_returns_all_same_page_regions_in_order():
    candidates = _candidate_screenshots(SCREENSHOTS, source_page=1)
    assert [c["path"] for c in candidates] == ["p1_a.png", "p1_b.png"]


def test_candidate_screenshots_falls_back_to_nearest_page():
    candidates = _candidate_screenshots(SCREENSHOTS, source_page=3)
    assert [c["path"] for c in candidates] == ["p2_a.png"]


def test_choose_screenshot_still_returns_a_single_best_guess():
    chosen = _choose_screenshot(SCREENSHOTS, source_page=1, action="Click Add.")
    assert chosen["path"] == "p1_a.png"


def test_explanation_never_calls_vision_and_uses_first_candidate(monkeypatch):
    # Explanations should not trigger a vision call even when several
    # screenshots exist on the page.
    calls = []

    def fake_vision_target(path, action):
        calls.append(path)
        return {"found": False, "target_name": "", "click_point": None,
                "bounding_box": None, "confidence": 0.0}

    monkeypatch.setattr(
        "app.services.ai_service._vision_target", fake_vision_target
    )

    screenshot, visual = _select_screenshot_and_visual(
        SCREENSHOTS, source_page=1, piece="Explains a concept.", kind="explanation"
    )

    assert screenshot["path"] == "p1_a.png"
    assert visual["found"] is False
    assert calls == []


def test_action_grounds_against_every_same_page_candidate_and_keeps_best(monkeypatch):
    # Regression test for the "always screenshot[0]" bug: the control
    # for this action is only visible in the SECOND screenshot on the
    # page, so grounding must pick that one, not the first.
    def fake_vision_target(path, action):
        if path == "p1_b.png":
            return {
                "found": True,
                "target_name": "Filter button",
                "click_point": [0.5, 0.5],
                "bounding_box": [0.4, 0.4, 0.6, 0.6],
                "confidence": 0.91,
            }
        return {"found": False, "target_name": "", "click_point": None,
                "bounding_box": None, "confidence": 0.0}

    monkeypatch.setattr(
        "app.services.ai_service._vision_target", fake_vision_target
    )

    screenshot, visual = _select_screenshot_and_visual(
        SCREENSHOTS, source_page=1, piece="Click Filter.", kind="action"
    )

    assert screenshot["path"] == "p1_b.png"
    assert visual["found"] is True
    assert visual["target_name"] == "Filter button"


def test_action_with_no_grounded_candidate_falls_back_to_first_screenshot(monkeypatch):
    monkeypatch.setattr(
        "app.services.ai_service._vision_target",
        lambda path, action: {
            "found": False, "target_name": "", "click_point": None,
            "bounding_box": None, "confidence": 0.0,
        },
    )

    screenshot, visual = _select_screenshot_and_visual(
        SCREENSHOTS, source_page=1, piece="Click Save.", kind="action"
    )

    assert screenshot["path"] == "p1_a.png"
    assert visual["found"] is False


# ----------------------------------------------------------------------
# Scene pacing / minimum duration floor
# ----------------------------------------------------------------------

def test_short_action_narration_is_padded_up_to_the_floor(tmp_path):
    import wave

    wav = tmp_path / "short.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        # ~0.3s clip -- shorter than the action floor.
        w.writeframes(b"\x00\x00" * int(16000 * 0.3))

    scene = {"kind": "action", "narration": "Click Save."}
    duration = _audio_duration_for_scene(scene, index=0, audio_paths=[str(wav)])

    assert duration >= MIN_SCENE_SECONDS["action"]


def test_long_action_narration_keeps_its_own_length_plus_padding(tmp_path):
    import wave

    wav = tmp_path / "long.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 5)  # 5s clip

    scene = {"kind": "action", "narration": "..."}
    duration = _audio_duration_for_scene(scene, index=0, audio_paths=[str(wav)])

    assert duration == 5.0 + SCENE_PADDING_SECONDS
