import numpy as np
import mujoco

from rl.model_utils import load_model, indices
from rl import config as C


def hold_test(gravcomp, seconds=2.0):
    
    """
    Test the robot's ability to hold a position for a given time with or without gravity compensation.
    Returns the maximum joint position error and whether all joint positions are finite.
    """

    model = load_model(gravcomp=gravcomp)
    data = mujoco.MjData(model)
    ix = indices(model)
    data.ctrl[ix["act"]] = 0.0
    
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    
    return float(np.max(np.abs(data.qpos[ix["qadr"]]))), bool(np.all(np.isfinite(data.qpos)))


# def tracking_test():
    
#     """
#     Test the robot's ability to track a target position.
#     Returns the time to settle, maximum velocity, and final error.
#     """
    
#     model = load_model()
#     data = mujoco.MjData(model)
#     ix = indices(model)
#     target = np.clip(np.array([0.5, -0.5, 0.5]), ix["low"], ix["high"])
#     data.ctrl[ix["act"]] = target
#     t_settle, vmax, err = None, 0.0, None
    
#     for k in range(int(3.0 / model.opt.timestep)):
        
#         mujoco.mj_step(model, data)
#         err = float(np.max(np.abs(data.qpos[ix["qadr"]] - target)))
#         vmax = max(vmax, float(np.max(np.abs(data.qvel[ix["dadr"]]))))
        
#         if t_settle is None and err < 0.03:
#             t_settle = (k + 1) * model.opt.timestep
    
#     return t_settle, vmax, err

def tracking_test():
    """
    Test target tracking and verify that the joint error remains
    within the settling tolerance for a sustained period.
    Returns settling time, maximum velocity, final error,
    maximum error, and settling status.
    """

    model = load_model()
    # Inspect body masses in the loaded model
    print("\nBody masses:")
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        print(f"{name}: {model.body_mass[body_id]:.3f} kg")

    print("Default geom density:", model.geom_size.shape)
    print("Total model mass:", model.body_mass.sum())

    print("\nMesh and geom details:")
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, gid
        )

    print(
        f"{name}: "
        f"type={model.geom_type[gid]}, "
        f"size={model.geom_size[gid]}, "
        f"mass contribution is included in its body"
    )

    data = mujoco.MjData(model)
    ix = indices(model)

    print("Loaded model:", C.TRAIN_MODEL)
    print("Timestep:", model.opt.timestep)

    for i, name in enumerate(C.JOINTS):
        jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        aid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, C.ACTUATORS[i]
        )

        print(
        f"{name}: "
        f"joint_range={model.jnt_range[jid]}, "
        f"actuator_ctrlrange={model.actuator_ctrlrange[aid]}, "
        f"kp={model.actuator_gainprm[aid, 0]}, "
        f"kv={model.actuator_biasprm[aid, 2]}"
        )

    target = np.clip(
        np.array([0.5, -0.5, 0.5]),
        ix["low"],
        ix["high"]
    )
    data.ctrl[ix["act"]] = target

    print("Requested target:", [0.5, -0.5, 0.5])
    print("Clipped target:", target)
    print("Control ranges:", model.actuator_ctrlrange[ix["act"]])
    print("Actuator gains:", model.actuator_gainprm[ix["act"]])

    tolerance = 0.03
    settle_duration = 0.1
    required_steps = max(1, int(np.ceil(settle_duration / model.opt.timestep)))

    t_settle = None
    vmax = 0.0
    err = None
    max_error = 0.0
    stable_steps = 0

    trace = []

    # for k in range(int(3.0 / model.opt.timestep)):
    for k in range(int(5.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if k % 50 == 0:
            trace.append((
                (k + 1) * model.opt.timestep,
                float(data.qpos[ix["qadr"][0]]),
                float(data.qvel[ix["dadr"][0]]),
            ))

        err = float(np.max(np.abs(data.qpos[ix["qadr"]] - target)))
        max_error = max(max_error, err)
        vmax = max(vmax, float(np.max(np.abs(data.qvel[ix["dadr"]]))))

        # Require the error to stay within tolerance for consecutive steps.
        if err < tolerance:
            stable_steps += 1
            if t_settle is None and stable_steps >= required_steps:
                t_settle = (k + 1 - required_steps + 1) * model.opt.timestep
        else:
            stable_steps = 0

    settled = t_settle is not None

    print("\nJoint 1 trace: time, position, velocity")
    for row in trace:
        print(f"{row[0]:.2f}s  q={row[1]:.4f} rad  "
            f"v={row[2]:.4f} rad/s")

    print("Final qpos:", data.qpos[ix["qadr"]])
    print("Final qvel:", data.qvel[ix["dadr"]])
    print("Final actuator forces:", data.actuator_force[ix["act"]])
    print("Final actuator controls:", data.ctrl[ix["act"]])
    print("Joint actuator force limits:",
      model.actuator_forcerange[ix["act"]])
    
    print("DOF armature:", model.dof_armature[ix["dadr"]])
    print("DOF friction loss:", model.dof_frictionloss[ix["dadr"]])
    print("Actuator bias parameters:",
      model.actuator_biasprm[ix["act"]])
    print("Actuator dynamics:",
      model.actuator_dyntype[ix["act"]])

    print("Joint damping:",
      model.dof_damping[ix["dadr"]])

    # Print the final joint positions and velocities
    print("Target positions:", target)
    print("Final joint positions:", data.qpos[ix["qadr"]])
    print("Final joint velocities:", data.qvel[ix["dadr"]])
    print("Final joint errors:", data.qpos[ix["qadr"]] - target)

    # Print actuator forces at the final simulation step
    print("Actuator forces:", data.actuator_force[ix["act"]])

    print("DOF body IDs:", model.dof_bodyid[ix["dadr"]])
    print("Body inertias:")
    for body_id in model.dof_bodyid[ix["dadr"]]:
        print(
            mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)
            ),
            "mass =", model.body_mass[body_id],
            "inertia =", model.body_inertia[body_id],
            "ipos =", model.body_ipos[body_id],
        )

    return t_settle, vmax, err, max_error, settled

model = load_model()
ix = indices(model)

print(f"Joint: {len(ix['qadr'])} | actuators: {len(ix['act'])} | Total mass: {model.body_mass.sum():.1f} kg")
print("Range joint [rad]:", [(round(float(a), 2), round(float(b), 2)) for a, b in zip(ix["low"], ix["high"])])

for gc in (False, True):
    err, ok = hold_test(gc)
    print(f"Holding 2s, gravcomp={gc}: errore max {err:.3f} rad, valori finiti={ok}")

# t, vmax, err = tracking_test()
t, vmax, err, max_error, settled = tracking_test()
# print(f"Tracking: Assested a 0.03 rad in {t} s | max velocity {vmax:.1f} rad/s | final error {err:.4f} rad")
print(f"Tracking settled within 0.03 rad: {settled}")
print(f"Settling time: {t if t is not None else 'not settled'} s")
print(f"Maximum velocity: {vmax:.1f} rad/s")
print(f"Final error: {err:.4f} rad")
print(f"Maximum tracking error: {max_error:.4f} rad")

if t is None or t >= 1.5:
    print("FAIL: The robot did not settle within 1.5 seconds.")
elif vmax >= 8.0:
    print("FAIL: Maximum joint velocity exceeded 8 rad/s.")
else:
    print("PASS: Settling time and maximum velocity requirements met.")