import numpy as np

class PDController:
    """
    Proportional-Derivative (PD) controller for robots with rotary joints.
    
    The controller calculates the control signal u for each joint:
    u[i] = Kp[i] * (q_target[i] - q[i]) + Kd[i] * (0 - qd[i])
    
    where:
    - q_target: target position of the joint [rad]
    - q: current joint position [rad]  
    - qd: current joint velocity [rad/s]
    - Kp: proportional gain
    - Kd: derivative gain
    """

    def __init__(self, kp, kd, n_joints):
        """
        Initializes the PD controller.
        
        Args:
            kp: proportional gain — can be a scalar or an array
            If a scalar, it is applied equally to all joints
            If an array, it must have length n_joints
            kd: derivative gain — same logic as kp
            n_joints: number of controlled joints
        """
        self.n_joints = n_joints

        # Convert to a NumPy array if scalar
        # This allows passing both Kp=100 and Kp=[100, 80]
        if np.isscalar(kp):
            self.kp = np.ones(n_joints) * kp
        else:
            self.kp = np.array(kp, dtype=float)

        if np.isscalar(kd):
            self.kd = np.ones(n_joints) * kd
        else:
            self.kd = np.array(kd, dtype=float)

        # Current target — initialized to zero
        self.q_target = np.zeros(n_joints)

        # Error history for analysis and debugging
        self.error_history = []

    def set_target(self, q_target):
        """
        Sets the target positions for all joints.
        
        Args:
            q_target: array of target positions [rad] must have length n_joints
        """
        q_target = np.array(q_target, dtype=float)

        # check if the length of targets position are equal to the number of joints
        if len(q_target) != self.n_joints:
            raise ValueError(
                f"q_target has {len(q_target)} elements "
                f"but the robot has {self.n_joints} joints"
            )

        self.q_target = q_target.copy() # copy to avoid accidental modifications from outside the class

    def compute(self, q, qd):
        """
        Calculates the PD control signal.
        
        Args:
            q: current joint positions [rad] — from data.qpos
            qd: current joint velocities [rad/s] — from data.qvel
            
        Returns:
        control_joint: control signal for each actuator to be assigned to data.ctrl
        """
        q = np.array(q, dtype=float)
        qd = np.array(qd, dtype=float)

        # Position error — how far we are from the target
        # Positive if the target is “ahead,” negative if it is “behind”
        position_error = self.q_target - q

        # Velocity error — we want zero velocity at the target
        # We use -qd directly because the velocity target is 0 (the target position is constant so the derivative is zero)
        velocity_error = -qd

        # PD control calculation (formula)
        # Element-wise operation: each joint has its own control
        control_joint = self.kp * position_error + self.kd * velocity_error

        # Save the error for further analysis or plotting
        self.error_history.append({
            'position_error': position_error.copy(),
            'velocity_error': velocity_error.copy(),
            'control_joint': control_joint.copy()
        })

        return control_joint

    def get_position_error(self, q):
        """
        Returns the current position error.
        Useful for checking whether the robot has reached the target.
        
        Args:
            q: current joint positions [rad]
            
        Returns:
            error: error for each joint [rad]
        """
        return self.q_target - np.array(q)

    def is_at_target(self, q, tolerance=0.01):
        """
        Checks whether the robot has reached the target.
        
        Args:
            q: current joint positions [rad]
            tolerance: acceptable error threshold [rad] - default 0.01 rad ≈ 0.57 degrees
                       
        Returns:
            True if all joints are within tolerance
        """
        error = np.abs(self.get_position_error(q))
        return np.all(error < tolerance)

    def reset(self):
        """Resets the controller — clears the error history."""
        self.error_history = [] # reset the error history 
        self.q_target = np.zeros(self.n_joints) # reset the target position to zero
