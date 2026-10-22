import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from rl import config as C
from rl.model_utils import load_model, indices

# Define a class , inerithing from gym.Env, implementing an environment to work simulation
class ArduinobotReachEnv(gym.Env):

    """
    Reaching 3D with joints 1-3 of arduinobot. 
    Action: delta position of joints
    """

    # Define the support to RGB array for robot visualzation or video recording
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, model_path=None, render_mode=None):
        super().__init__()

        # ---------- import model and set parameters of the environment and simulation data ----------
        self.model = load_model(model_path)
        self.data = mujoco.MjData(self.model)
        self.ix = indices(self.model)
        self.render_mode = render_mode
        self._renderer = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        high = np.full(15, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in C.MARKER_BODIES]
        self.marker_ids = [i for i in ids if i >= 0]
        self.pool = self._build_pool()
        self.target = self.pool[0].copy()
        self.ctrl_target = np.zeros(3)
        self.steps = 0

    # ---------- target ----------
    def _build_pool(self):

        """Reachable targets: positions of the end effector with random joints."""

        d = mujoco.MjData(self.model)
        rng = np.random.default_rng(0)
        pts = np.empty((C.POOL_SIZE, 3))

        for i in range(C.POOL_SIZE):
            # Genrate random joint positions within the limits and compute the end effector position and update it
            d.qpos[self.ix["qadr"]] = rng.uniform(self.ix["low"], self.ix["high"])
            mujoco.mj_kinematics(self.model, d)
            pts[i] = d.xpos[self.ix["ee"]]

        # Filter the points that are within the target box (defined in config.py)
        m = ((pts[:, 0] >= C.TARGET_X[0]) & 
             (pts[:, 0] <= C.TARGET_X[1]) &
             (pts[:, 1] >= C.TARGET_Y[0]) & 
             (pts[:, 1] <= C.TARGET_Y[1]) &
             (pts[:, 2] >= C.TARGET_Z[0]) & 
             (pts[:, 2] <= C.TARGET_Z[1]))
        pool = pts[m]
        
        # Return an error if the pool is too small (review the target)
        if len(pool) < C.MIN_POOL:
            raise RuntimeError(
                f"Target pool too small ({len(pool)}): review TARGET_X/Y/Z in rl/config.py "
                "using the results of workspace_analysis.py")
        return pool

    # Move the markers to the target position in the simulation model
    def _place_markers(self):
        for i in self.marker_ids:
            self.model.body_pos[i] = self.target
        mujoco.mj_forward(self.model, self.data)

    # ---------- observation ----------
    def _obs(self):
        q = self.data.qpos[self.ix["qadr"]]
        qd = self.data.qvel[self.ix["dadr"]]
        ee = self.data.xpos[self.ix["ee"]]
        return np.concatenate([q, qd, ee, self.target, self.target - ee]).astype(np.float32)

    # Measure the euclidean distance between the end effector and the target
    def _dist(self):
        return float(np.linalg.norm(self.target - self.data.xpos[self.ix["ee"]]))

    # ---------- API Gymnasium ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the simulation data
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)
        
        # Set the initial joint positions to random values within a small range around zero
        q0 = self.np_random.uniform(-0.1, 0.1, size=3)
        self.data.qpos[self.ix["qadr"]] = q0
        self.ctrl_target = q0.copy()
        self.data.ctrl[self.ix["act"]] = self.ctrl_target

        # Pick the target position from the pool or from the options provided
        if "target" in options:
            # External target position (from options)
            self.target = np.asarray(options["target"], dtype=float).copy()
        else:
            # Random target position (from pool)
            self.target = self.pool[self.np_random.integers(len(self.pool))].copy()
        # Update the kinematics 
        mujoco.mj_forward(self.model, self.data)

        # Position the markers and set the step counter to zero
        if options.get("move_markers", True):
            self._place_markers()
        self.steps = 0

        return self._obs(), {"distance": self._dist()}

    def step(self, action):
        
        '''
        Clip the action to be within the range [-1, 1] 
        and update the control target based on the action and maximum delta allowed. 
        Then, perform multiple substeps of simulation to update the state of the environment. 
        After that, compute the distance to the target, 
        determine if the task is successful, 
        calculate the reward, 
        and check if the episode is terminated or truncated 
        based on success or maximum steps reached. 
        Finally, return the observation, reward, 
        termination status, truncation status, and additional info.
        '''

        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.ctrl_target = np.clip(self.ctrl_target + a * C.MAX_DELTA, self.ix["low"], self.ix["high"])
        self.data.ctrl[self.ix["act"]] = self.ctrl_target
        
        for _ in range(C.N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        
        mujoco.mj_forward(self.model, self.data)   # xpos updated at the current state
        self.steps += 1

        dist = self._dist()
        success = dist < C.SUCCESS_RADIUS
        reward = -dist - 0.01 * float(np.sum(a ** 2)) + (5.0 if success else 0.0)
        terminated = bool(success)
        truncated = self.steps >= C.MAX_STEPS
        info = {"distance": dist, "is_success": bool(success)}
        
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        
        """ 
        Define the rendering of the environment using mujoco's renderering capabilities."""

        if self.render_mode != "rgb_array":
            return None
        
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=C.IMG_H, width=C.IMG_W)
        
        # Update the scene with the current simulation data and render it from the side camera view
        self._renderer.update_scene(self.data, camera="side_cam")
        
        return self._renderer.render()

    def close(self):
        
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None