import os

MODEL_PATH = os.path.expanduser("~/mujoco_robotics/models/arduinobot_vision.xml")
CAMERA = "top_cam"
CUBE_BODY = "target_cube"
CUBE_Z = 0.04            # height of the cube center in meters (4 cm)
IMG_W, IMG_H = 640, 480

# Magenta in HSV of OpenCV (H goes from 0 to 179)
HSV_LOWER = (135, 100, 50)
HSV_UPPER = (165, 255, 255)
MIN_AREA = 50  # pixel, minimum area of the detected contour to be considered a cube
KERNEL_SIZE = 5