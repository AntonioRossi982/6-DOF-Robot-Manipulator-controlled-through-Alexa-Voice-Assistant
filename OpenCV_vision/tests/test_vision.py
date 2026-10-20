import numpy as np
import pytest

from vision.scene import load_scene, place_cube, render
from vision.cube_detector import detect_cube
from vision.projection import pixel_to_world
from vision import config

MAX_ERR_M = 0.015  # 15 mm (tolerance)


@pytest.fixture(scope="module")
def scene():
    model, data, renderer, cube_id = load_scene()
    yield model, data, renderer, cube_id
    renderer.close()


@pytest.mark.parametrize("x,y", [(0.30, 0.40), (0.50, 0.60), (0.45, 0.35)])
def test_position_error_below_threshold(scene, x, y):
    model, data, renderer, cube_id = scene
    true = place_cube(model, data, cube_id, x, y)
    det = detect_cube(render(renderer, data))
    assert det is not None
    est = pixel_to_world(model, data, config.CAMERA, det["u"], det["v"],
                         config.IMG_W, config.IMG_H, config.CUBE_Z)
    assert np.linalg.norm(est[:2] - true[:2]) < MAX_ERR_M


def test_no_false_positive_when_cube_out_of_view(scene):
    model, data, renderer, cube_id = scene
    place_cube(model, data, cube_id, 5.0, 5.0)
    assert detect_cube(render(renderer, data)) is None