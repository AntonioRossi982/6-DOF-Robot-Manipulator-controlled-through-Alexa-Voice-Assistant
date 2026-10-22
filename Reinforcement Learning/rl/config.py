import os

# --- paths ---
ROOT = os.path.expanduser("~/mujoco_robotics")
TRAIN_MODEL = os.path.join(ROOT, "models", "arduinobot_train.xml")
RL_VISION_MODEL = os.path.join(ROOT, "models", "arduinobot_rl_vision.xml")
RUNS_DIR = os.path.join(ROOT, "runs")
MODELS_DIR = os.path.join(ROOT, "rl_models")
DOCS_DIR = os.path.join(ROOT, "docs")

# --- robot configuration ---
JOINTS = ["joint_1", "joint_2", "joint_3"]
ACTUATORS = ["act_joint1", "act_joint2", "act_joint3"]
EE_BODY = "end_effector"
MARKER_BODIES = ["target_marker", "target_cube"]

# --- dynamics and control ---
N_SUBSTEPS = 25           # 25 x 0.002 s = 0.05 s -> 20 Hz of control frequency
MAX_DELTA = 0.08          # rad per step of control frequency an |action| = 1
MAX_STEPS = 100           # 5 s per episode
SUCCESS_RADIUS = 0.03     # in meters
GRAVCOMP = True           # simplification: gravity compensated on links
FLOOR_ONLY_CONTACTS = True  # simplification: only contact with the floor

# --- target zone ---
# Evaluate the x0, x1, y0, y1, zc values in the XML file of the model to set the target zone
# TARGET_X = (0.2 - 0.05, 0.60 + 0.05) # x0 = 0.25, x1 = 0.60
# TARGET_Y = (0.30 - 0.05, 0.80 + 0.05) # y0 = 0.30, y1 = 0.80
# TARGET_Z = (0.04 - 0.10, 0.04 + 0.10) # zc = 0.04, height of the cube center in meters (4 cm)
# TARGET_X = (-0.18, -0.12)
# TARGET_Y = (1.04, 1.10)
# TARGET_Z = (1.30, 1.42)
TARGET_X = (-0.25, 0.25)
TARGET_Y = (0.15, 0.45)
TARGET_Z = (0.30, 0.50)

POOL_SIZE = 60_000        # configurations sampled to build the pool of reachable targets
MIN_POOL = 300

# --- Camera vision (copy from models/arduinobot_vision.xml) ---
CAM_POS = (0.4, 0.6, 1.8)
CAM_FOVY = 45.0
IMG_W, IMG_H = 640, 480