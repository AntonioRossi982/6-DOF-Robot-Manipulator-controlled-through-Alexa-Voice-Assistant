import os
os.environ.setdefault("MUJOCO_GL", "egl")  # Alternative you can use "osmesa" instead of "egl" if you have issues with EGL
import mujoco
import imageio.v2 as imageio

MODEL = os.path.expanduser("~/mujoco_robotics/models/arduinobot_vision.xml")
OUT = os.path.expanduser("~/mujoco_robotics/data/frames")
os.makedirs(OUT, exist_ok=True)

model = mujoco.MjModel.from_xml_path(MODEL) # load the scene
data = mujoco.MjData(model) # load the current state of the simulation
mujoco.mj_forward(model, data) # evaluate the kinematics and dynamics from the current state (geometry pose and geometries)

renderer = mujoco.Renderer(model, height=480, width=640) # create a renderer for the scene
try:
    renderer.update_scene(data, camera="top_cam") # check the scene from the camera defined in the XML file
    frame = renderer.render() # Generate the frame (expected 480, 640, 3) with the RGB values of the scene
finally:
    renderer.close() # close the renderer even if the EGL returned an error

imageio.imwrite(os.path.join(OUT, "frame_000.png"), frame) # write the array in a PNG file
print(frame.shape, frame.dtype) # print shape and data type of the frame (expected (480, 640, 3) and uint8)