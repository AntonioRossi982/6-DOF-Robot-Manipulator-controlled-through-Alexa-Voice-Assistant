import mujoco
import numpy as np

def load_model(xml_path):
    
    """
    Load an MJCF model from a file and generate the simulation data.
    
    In MuJoCo, there are two fundamental objects:
    - MjModel: describes the robot's static structure (geometry, 
      masses, joints). It does not change during the simulation.
    - MjData: contains the current dynamic state (positions, 
      velocities, forces). It changes at every step.
    """

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data

def get_joint_positions(data):
    """
    Returns the current positions of all joints.
    
    data.qpos is the vector of generalized positions.
    For robots with only hinge joints, qpos[i] 
    is the angle in radians of joint i.
    """
    return data.qpos.copy()

def get_joint_velocities(data):
    """
    Returns the current velocities of all joints.
    
    data.qvel is the vector of generalized velocities.
    """
    return data.qvel.copy()

def get_body_position(model, data, body_name):
    """
    Returns the 3D position of a body in the world frame.
    
    data.xpos[body_id] is the position of the body's center
    in the world frame after the last simulation step.
    """
    body_id = model.body(body_name).id
    return data.xpos[body_id].copy()

def set_control(data, ctrl_values):
    """
    Set the control signal for all actuators.
    
    data.ctrl is the vector of control signals.
    data.ctrl[i] corresponds to actuator i defined in <actuator>.
    """
    data.ctrl[:] = ctrl_values

def simulation_step(model, data):
    """
    Advance the simulation by one timestep.
    
    mj_step does everything: 
    1. Apply controls (data.ctrl)
    2. Calculates the forces (gravity, contacts, actuators)
    3. Integrates the equations of motion
    4. Updates positions and velocities
    5. Updates the positions of the bodies in the world (xpos, xmat)
    """
    mujoco.mj_step(model, data)


