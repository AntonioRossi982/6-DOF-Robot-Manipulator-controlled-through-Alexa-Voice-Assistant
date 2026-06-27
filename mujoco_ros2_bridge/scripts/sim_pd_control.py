import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco
import mujoco.viewer
import numpy as np
import time
from utils.mujoco_utils import * 
from controllers.pd_controller import PDController

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, '2dof_arm.xml')

# Sequence of targets to be reached one after another (sequence of joint angles in radians)
# Each target is [joint1_rad, joint2_rad]
TARGETS = [
    [0.5,  0.3],    # Target 1 — arm to the right and up
    [-0.5, 0.3],    # Target 2 — arm to the left and up
    [0.0, -0.5],    # Target 3 — arm centered and down
    [0.8,  0.8],    # Target 4 — maximum right extension
    [0.0,  0.0],    # Target 5 — home position
]

def print_state(sim_time, q, qd, ee_pos, control_joint, error, at_target):
    """Prints the current state in a readable format."""
    print(f"\n{('='*50)}")
    print(f"Simulation time: {sim_time:.2f}s")
    print(f"Joint positions:   j1={q[0]:+.3f} rad  j2={q[1]:+.3f} rad")
    print(f"Joint velocities:    j1={qd[0]:+.3f} r/s  j2={qd[1]:+.3f} r/s")
    print(f"End effector:      x={ee_pos[0]:+.3f}  y={ee_pos[1]:+.3f}  z={ee_pos[2]:+.3f} m")
    print(f"Control:         act1={control_joint[0]:+.3f}  act2={control_joint[1]:+.3f}")
    print(f"Position error:  e1={error[0]:+.3f} rad  e2={error[1]:+.3f} rad")
    print(f"Target reached:  {'✓ YES' if at_target else '✗ NO'}")

def main():
    model, data = load_model(MODEL_PATH)

    # Initialize the PD controller
    # Kp=150: proportional gain — higher = faster response
    # Kd=15: derivative gain — higher = more damping
    # Starting rule of thumb: Kd ≈ Kp / 10
    controller = PDController(
        kp=150.0, # other choices; 150.0, 150.0, 200.0, 150.0, 150.0
        kd=15.0,  # other choices: 15.0, 5.0, 0.0, 80.0, 15.0
        n_joints=2
    )

    # Index of the current target in the sequence
    current_target_idx = 0
    controller.set_target(TARGETS[current_target_idx])

    print(f"PD controller initialized")
    print(f"Kp={controller.kp}, Kd={controller.kd}")
    print(f"First target: {TARGETS[current_target_idx]} rad\n")

    # Minimum time to spend on each target before moving on to the next one
    # Even if the target is reached earlier, wait this amount of time
    time_on_target = 0.0
    min_time_on_target = 2.0  # seconds

    sim_time = 0.0
    last_print_time = 0.0
    print_interval = 0.5  # print every half second

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"Simulation started.\n")

        while viewer.is_running():
            step_start = time.time()

            # === READ THE STATE ===
            q = get_joint_positions(data)
            qd = get_joint_velocities(data)
            ee_pos = get_body_position(model, data, 'end_effector')

            # === COMPUTE THE PD CONTROL ===
            control_joint = controller.compute(q, qd)

            # Limit the control to the range defined in the MJCF [-1, 1]
            # np.clip clips values outside the range
            control_joint = np.clip(control_joint, -1.0, 1.0)

            # === APPLY CONTROL ===
            set_control(data, control_joint)

            # === CHECK TARGET ===
            error = controller.get_position_error(q)
            at_target = controller.is_at_target(q, tolerance=0.02)

            if at_target:
                time_on_target += model.opt.timestep
            else:
                time_on_target = 0.0

            # Move to the next target if:
            # 1. The target has been reached
            # 2. We have been at the target for at least min_time_on_target seconds
            if at_target and time_on_target >= min_time_on_target:
                current_target_idx = (current_target_idx + 1) % len(TARGETS)
                controller.set_target(TARGETS[current_target_idx])
                time_on_target = 0.0
                print(f"\n>>> New target [{current_target_idx}]: "
                      f"{TARGETS[current_target_idx]} rad")

            # === ADVANCE THE SIMULATION ===
            simulation_step(model, data)
            sim_time += model.opt.timestep

            # === PRINT THE STATE ===
            if sim_time - last_print_time >= print_interval:
                last_print_time = sim_time
                print_state(sim_time, q, qd, ee_pos, control_joint, error, at_target)

            # === UPDATE VIEWER ===
            viewer.sync()

            # === MAINTAIN TIMESTEP ===
            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining) # sleep to maintain real-time pace (no GPU resources lost)

if __name__ == '__main__':
    main()