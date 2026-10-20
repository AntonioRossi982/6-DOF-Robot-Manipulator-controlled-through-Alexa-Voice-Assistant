import os
os.environ.setdefault("MUJOCO_GL", "egl")  # Before MuJoCo import
import mujoco
from vision import config


def load_scene():
    model = mujoco.MjModel.from_xml_path(config.MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=config.IMG_H, width=config.IMG_W)
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.CUBE_BODY)
    return model, data, renderer, cube_id


def place_cube(model, data, cube_id, x, y):
    """Move the cube and return its world position."""
    model.body_pos[cube_id] = [x, y, config.CUBE_Z]
    mujoco.mj_forward(model, data)
    return data.xpos[cube_id].copy()


def render(renderer, data):
    renderer.update_scene(data, camera=config.CAMERA)
    return renderer.render()