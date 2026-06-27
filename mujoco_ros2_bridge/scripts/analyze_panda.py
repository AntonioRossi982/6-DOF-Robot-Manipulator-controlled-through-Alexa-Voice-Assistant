import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco
import numpy as np

# Path to the Panda model
PANDA_PATH = os.path.expanduser('~/mujoco_menagerie/franka_emika_panda/panda.xml')


def analyze_model(model):
    """
    Analyzes and prints the model's main properties.
    Essential for understanding how to control a new robot.
    """
    print("=" * 60)
    print("MODEL ANALYSIS: Franka Emika Panda")
    print("=" * 60)

    # General properties
    print(f"\n--- General properties ---")
    print(f"Number of bodies:      {model.nbody}")
    print(f"Number of joints:     {model.njnt}")
    print(f"Number of DOFs:       {model.nv}")
    print(f"Number of actuators: {model.nu}")
    print(f"Time step:            {model.opt.timestep} s")

    # List of joints with properties
    print(f"\n--- Joint ---")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, i
        )
        joint_type = model.jnt_type[i]

        # Joint type: 0=free, 1=ball, 2=slide, 3=hinge
        type_names = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}
        type_str = type_names.get(joint_type, 'unknown')

        # Joint limits
        limited = model.jnt_limited[i]
        if limited:
            low = model.jnt_range[i, 0]
            high = model.jnt_range[i, 1]
            range_str = f"[{low:.3f}, {high:.3f}] rad"
        else:
            range_str = "unlimited"
        
        print(f"  Joint {i}: {joint_name:20s} type={type_str:6s} range={range_str}")

    # List of actuators
    print(f"\n--- Actuators ---")
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
        )
        # Control signal limits
        if model.actuator_ctrllimited[i]:
            low = model.actuator_ctrlrange[i, 0]
            high = model.actuator_ctrlrange[i, 1]
            ctrl_str = f"[{low:.1f}, {high:.1f}]"
        else:
            ctrl_str = "unlimited"

        print(f"  Actuator {i}: {act_name:20s} ctrl_range={ctrl_str}")

    # List of bodies with masses
    print(f"\n--- Bodies and masses ---")
    total_mass = 0.0
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, i
        )
        mass = model.body_mass[i]
        total_mass += mass
        if mass > 0.001:  # print only bodies with significant mass
            print(f"  Body {i}: {body_name:25s} mass={mass:.3f} kg")

    print(f"\n  Estimated total mass: {total_mass:.3f} kg")
    print("=" * 60)


def main():
    model = mujoco.MjModel.from_xml_path(PANDA_PATH)
    analyze_model(model)

if __name__ == '__main__':
    main()