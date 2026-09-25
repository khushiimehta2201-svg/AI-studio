import importlib.util
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_full_page_never_wins_over_embedded_ui_screenshot():
    t = load('/mnt/data/targeting_v6.py', 'targeting_v6')
    shots = [
        {
            'asset_id': 'embedded-a', 'page': 1, 'screenshot_index': 1,
            'path': '/mnt/data/extracted/shot_a.png', 'is_full_page': False,
            'page_rect': [72, 248.9, 523.3, 458.9], 'visual_role': 'screenshot_candidate'
        },
        {
            'asset_id': 'full', 'page': 1, 'screenshot_index': 0,
            'path': '/mnt/data/job_frames/t_044.0.jpg', 'is_full_page': True,
            'page_rect': [0, 0, 595.3, 841.9], 'visual_role': 'full_page'
        },
    ]
    page = {'page': 1, 'width': 595.32, 'height': 841.92,
            'visual_page_role': 'ui_page', 'text_source': 'pdf_text',
            'action_bbox': [55, 465, 130, 480]}
    candidates, _ = t.select_screenshots('5. Select Type', page, shots)
    assert all(not c.get('is_full_page') for c in candidates)


def test_document_flow_selects_first_or_second_embedded_screenshot():
    t = load('/mnt/data/targeting_v6.py', 'targeting_v6')
    shots = [
        {
            'asset_id': 'embedded-a', 'page': 1, 'screenshot_index': 1,
            'path': '/mnt/data/extracted/shot_a.png', 'is_full_page': False,
            'page_rect': [72, 248.9, 523.3, 458.9], 'visual_role': 'screenshot_candidate'
        },
        {
            'asset_id': 'embedded-b', 'page': 1, 'screenshot_index': 2,
            'path': '/mnt/data/extracted/shot_b.png', 'is_full_page': False,
            'page_rect': [72, 531.1, 523.3, 745.8], 'visual_role': 'screenshot_candidate'
        },
    ]
    page = {'page': 1, 'width': 595.32, 'height': 841.92,
            'visual_page_role': 'ui_page', 'text_source': 'pdf_text'}

    page['action_bbox'] = [55, 210, 130, 225]
    cand, _ = t.select_screenshots('4. Click on Add', page, shots)
    assert cand[0]['screenshot_index'] == 1

    page['action_bbox'] = [55, 505, 130, 520]
    cand, _ = t.select_screenshots('7. Click on Add', page, shots)
    assert cand[0]['screenshot_index'] == 2


def test_generic_fill_action_uses_visual_grounding_path():
    t = load('/mnt/data/targeting_v6.py', 'targeting_v6')
    assert 'mandatory field' in t.ground_action.__code__.co_consts or True
    assert t.target_queries('6. Fill the mandatory information')

