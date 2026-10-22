import mujoco
import numpy as np
from rl import config as C


def load_model(path=None, gravcomp=None, floor_only=None):
    
    """
    Load the MuJoCo model from XML file and apply simplifications.
    If path is None, load the default TRAIN_MODEL.
    If gravcomp is None, use the default C.GRAVCOMP.
    If floor_only is None, use the default C.FLOOR_ONLY_CONTACTS.
    """
    
    model = mujoco.MjModel.from_xml_path(path or C.TRAIN_MODEL)
    gravcomp = C.GRAVCOMP if gravcomp is None else gravcomp
    floor_only = C.FLOOR_ONLY_CONTACTS if floor_only is None else floor_only
    
    if gravcomp:
        model.body_gravcomp[:] = 1.0
    
    if floor_only:
        # the robot geometries don't collide with each other, but only with the floor
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for geometries in range(model.ngeom):
            if geometries != floor:
                model.geom_conaffinity[geometries] = 0
    
    return model


def indices(model):
    """Joints, actuators, and end-effector body indices in the MuJoCo model."""
    
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in C.JOINTS]
    aid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in C.ACTUATORS]
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, C.EE_BODY)
    
    assert min(jid + aid + [ee]) >= 0, "Names of joints, actuators, or end-effector body not found in the MuJoCo model."
    
    return {
        "qadr": model.jnt_qposadr[jid].copy(),
        "dadr": model.jnt_dofadr[jid].copy(),
        "low": model.jnt_range[jid, 0].copy(),
        "high": model.jnt_range[jid, 1].copy(),
        "act": np.array(aid),
        "ee": ee,
    }