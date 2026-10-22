import numpy as np
import mujoco
from rl import config as C


class DLSController:
    """Inverse kinematics controller using damped least squares."""

    def __init__(self, env, damping=0.05, gain=1.0):
        self.env = env
        self.damping = damping
        self.gain = gain
        self.Jacobian = np.zeros((3, env.model.nv))

    def action(self):
        env = self.env

        mujoco.mj_jacBody(
            env.model,
            env.data,
            self.Jacobian,
            None,
            env.ix["ee"],
        )

        J = self.Jacobian[:, env.ix["dadr"]]
        e = env.target - env.data.xpos[env.ix["ee"]]
        distance = np.linalg.norm(e)

        # Adaptive Cartesian gain: smaller steps near the target.
        if distance > 0.10:
            adaptive_gain = self.gain
        else:
            adaptive_gain = self.gain * max(0.2, distance / 0.10)

        dq = (
            J.T
            @ np.linalg.solve(
                J @ J.T + self.damping**2 * np.eye(3),
                e,
            )
        ) * self.gain

        # Limit the overall joint-space step, preserving its direction.
        max_step = C.MAX_DELTA
        norm_dq = np.linalg.norm(dq)
        if norm_dq > max_step:
            dq = dq * (max_step / norm_dq)

        return (dq / C.MAX_DELTA).astype(np.float32)

# Previous DLSController class with different dq formula and saturation
# class DLSController:
    
#     """
#     Inverse kinematics controller 
#     using damped least squares.
#     """

#     def __init__(self, 
#                  env, 
#                  damping=0.05, 
#                  gain=1.0):

#         self.env = env
#         self.damping = damping
#         self.gain = gain
#         self.Jacobian = np.zeros((3, env.model.nv))

#     def action(self):

#         env = self.env
#         mujoco.mj_jacBody(env.model, env.data, self.Jacobian, None, env.ix["ee"])
        
#         J = self.Jacobian[:, env.ix["dadr"]] # 3x3
#         e = env.target - env.data.xpos[env.ix["ee"]] # evalute the error: ideal position - end-effector position
#         # Damped least squares solution for the joint velocities (dq) to minimize the error in the end-effector position
#         dq = J.T @ np.linalg.solve(J @ J.T + self.damping ** 2 * np.eye(3), e) * self.gain
        
#         # Connect the kinematics controller to the environment by returning the joint velocities (dq) as the action
#         return np.clip(dq / C.MAX_DELTA, -1.0, 1.0).astype(np.float32)