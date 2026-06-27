import sys
import os

# import of folders in parent directory to access robot models in models folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco
import mujoco.viewer
import numpy as np
import time
from utils.mujoco_utils import (
    load_model,
    get_joint_positions,
    get_joint_velocities,
    get_body_position,
    set_control,
    simulation_step
)

# Absolute path to the 2dof_arm.xml model file (in models folder)
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, '2dof_arm.xml')

def main():
    # Load the model and create the simulation data
    model, data = load_model(MODEL_PATH)
    
    print("=== Model Information ===")
    print(f"Model name: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 0)}")
    print(f"Number of bodies (links): {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print(f"Number of DOFs: {model.nv}")
    print(f"Time step: {model.opt.timestep} s")
    print(f"Simulation frequency: {1/model.opt.timestep:.0f} Hz")
    print("================================\n")

    # Variables to track the simulation time and printing intervals
    sim_time = 0.0
    last_print_time = 0.0
    print_interval = 1.0  # print every 1 second of simulation time
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        print("Simulation started. Press Ctrl+C to exit.")
        print("Press SPACE in the viewer to pause/play.\n")
        
        start_wall_time = time.time() # Track the real wall-clock time to compute elapsed time for control signals
        
        while viewer.is_running():
            step_start = time.time() # Start time of the current viewer simulation
            
            # === READ THE CURRENT STATE (position, velocities and end effector position)===
            q = get_joint_positions(data)    # joint angles [rad]
            qd = get_joint_velocities(data)  # joint velocities [rad/s]
            ee_pos = get_body_position(model, data, 'end_effector')

            # === EVALUATE THE CONTROL SIGNAL ===
            # Sinusoidal signal to move the arm
            # Different frequencies for the two joints — complex motion
            t = time.time() - start_wall_time
            # There are 2 actuators, so we create an array of 2 control values
            ctrl = np.array([
                0.5 * np.sin(0.5 * t),       # joint1: slow joint
                0.3 * np.sin(1.0 * t)        # joint2: fast joint
            ])

            # === APPLY THE CONTROL SIGNAL ===
            set_control(data, ctrl)
            
            # === ADVANCE THE SIMULATION ===
            simulation_step(model, data)
            sim_time += model.opt.timestep
            
            # === PRINT THE STATE ===
            if sim_time - last_print_time >= print_interval:
                last_print_time = sim_time
                print(f"t={sim_time:.1f}s")
                print(f"  Joint positions: j1={q[0]:.3f} rad, j2={q[1]:.3f} rad")
                print(f"  Joint velocities: j1={qd[0]:.3f} rad/s, j2={qd[1]:.3f} rad/s")
                print(f"  End effector: x={ee_pos[0]:.3f} y={ee_pos[1]:.3f} z={ee_pos[2]:.3f} m")
                print(f"  Control: act1={ctrl[0]:.3f} act2={ctrl[1]:.3f}")
                print()
            
            # === UPDATE THE VIEWER ===
            viewer.sync()
            
            # === MAINTAIN THE REAL TIMESTEP ===
            # Avoid that the simulation goes faster than real time
            elapsed = time.time() - step_start # Time taken for the current simulation step
            remaining = model.opt.timestep - elapsed # Time left to maintain the real-time pace
            if remaining > 0:
                time.sleep(remaining) # Sleep to maintain the real-time pace of the simulation

if __name__ == '__main__':
    main()