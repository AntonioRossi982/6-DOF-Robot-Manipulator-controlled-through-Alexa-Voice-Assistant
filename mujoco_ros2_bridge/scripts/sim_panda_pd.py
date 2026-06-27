import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco
import mujoco.viewer
import numpy as np
import time
from controllers.pd_controller import PDController

PANDA_PATH = os.path.expanduser(
    '~/mujoco_menagerie/franka_emika_panda/panda.xml'
)


def get_arm_joints(model, data):
    """
    Extract only the positions of the 7 arm joints.

    The Panda has joints named 'joint1' ... 'joint7'.
    We need to find their indices in qpos to extract
    only the values we are interested in.

    Note: qpos may also contain other DOFs
    (for example the gripper fingers) — this is why
    we do not simply use data.qpos[:7].
    """
    arm_joint_names = [f'joint{i}' for i in range(1, 8)]
    positions = []
    velocities = []

    for name in arm_joint_names:
        joint_id = model.joint(name).id

        # qpos_adr is the joint address in qpos
        addr = model.jnt_qposadr[joint_id]
        positions.append(data.qpos[addr])

        # dof_adr is the address in qvel
        dof_addr = model.jnt_dofadr[joint_id]
        velocities.append(data.qvel[dof_addr])

    return np.array(positions), np.array(velocities)


def get_arm_actuator_indices(model):
    """
    Find the arm actuator indices in data.ctrl.

    The Panda has actuators named 'actuator1' ... 'actuator7'.
    Returns their indices so that we can write only
    those values into data.ctrl.
    """
    indices = []
    for i in range(1, 8):
        act_id = model.actuator(f'actuator{i}').id
        indices.append(act_id)
    return indices


def main():
    model = mujoco.MjData.__new__(mujoco.MjData)
    model = mujoco.MjModel.from_xml_path(PANDA_PATH)
    data = mujoco.MjData(model)

    # Find the arm actuator indices
    arm_actuator_idx = get_arm_actuator_indices(model)
    print(f"Arm actuator indices: {arm_actuator_idx}")

    # Target positions for the arm (7 joints)
    # Values are in radians and respect Panda limits
    TARGETS = [
        [0.0,  0.0,  0.0, -1.57, 0.0,  1.57, 0.0],   # home
        [0.5, -0.5,  0.0, -2.0,  0.0,  1.5,  0.5],   # position A
        [-0.5, 0.3,  0.3, -1.0,  0.0,  1.2, -0.5],   # position B
        [0.0,  0.0,  0.0, -1.57, 0.0,  1.57, 0.0],   # home
    ]

    # Initialize the PDController for 7 joints
    # Higher Kp and Kd compared to the 2DOF robot
    # because the Panda has more mass and inertia
    controller = PDController(
        kp=200.0,
        kd=20.0,
        n_joints=7
    )

    # Move the robot to the home position at startup
    controller.set_target(TARGETS[0])

    current_target_idx = 0
    time_at_target = 0.0
    min_time_at_target = 3.0
    sim_time = 0.0
    last_print_time = 0.0

    print("Panda simulation with PD controller")
    print(f"Kp={controller.kp[0]}, Kd={controller.kd[0]}")
    print(f"Initial target: {TARGETS[0]}\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Read arm state
            q, qd = get_arm_joints(model, data)

            # Compute PD control
            ctrl = controller.compute(q, qd)

            # Clamp to actuator range
            ctrl = np.clip(ctrl, -1.0, 1.0)

            # Apply control only to arm actuators
            for i, act_idx in enumerate(arm_actuator_idx):
                data.ctrl[act_idx] = ctrl[i]

            # Check target
            at_target = controller.is_at_target(q, tolerance=0.05) # almost 3 degrees from target

            if at_target:
                time_at_target += model.opt.timestep
            else:
                time_at_target = 0.0

            # Move to the next target
            if at_target and time_at_target >= min_time_at_target:
                current_target_idx = (current_target_idx + 1) % len(TARGETS)
                controller.set_target(TARGETS[current_target_idx])
                time_at_target = 0.0

                print(
                    f">>> New target [{current_target_idx}]: "
                    f"{TARGETS[current_target_idx]}"
                )

            # Advance simulation
            mujoco.mj_step(model, data)
            sim_time += model.opt.timestep

            # Print status every second
            if sim_time - last_print_time >= 1.0:
                last_print_time = sim_time
                error = controller.get_position_error(q)

                print(
                    f"t={sim_time:.1f}s | "
                    f"max_error={np.max(np.abs(error)):.4f} rad | "
                    f"target={'reached' if at_target else 'in progress'}"
                )

            viewer.sync() # update the viewer with the latest simulation state

            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining) # sleep to maintain real-time simulation


if __name__ == '__main__':
    main()