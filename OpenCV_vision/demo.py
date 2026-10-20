import os
import cv2
import numpy as np
import imageio.v2 as imageio

from vision.scene import load_scene, place_cube, render
from vision.cube_detector import detect_cube
from vision.projection import pixel_to_world
from vision import config

OUT = os.path.expanduser("~/mujoco_robotics/docs")
os.makedirs(OUT, exist_ok=True)

model, data, renderer, cube_id = load_scene()
rng = np.random.default_rng(1)
frames = []
for _ in range(12):
    x, y = rng.uniform(0.25, 0.60), rng.uniform(0.30, 0.80)
    true = place_cube(model, data, cube_id, x, y)
    frame = render(renderer, data)
    det = detect_cube(frame)
    if det is None:
        continue
    est = pixel_to_world(model, data, config.CAMERA, det["u"], det["v"],
                         config.IMG_W, config.IMG_H, config.CUBE_Z)
    err = np.linalg.norm(est[:2] - true[:2]) * 1000
    out = frame.copy()
    cv2.drawContours(out, [det["contour"]], -1, (0, 255, 0), 2)
    cv2.circle(out, (int(round(det["u"])), int(round(det["v"]))), 5, (255, 255, 255), -1)
    cv2.putText(out, f"est=({est[0]:.3f},{est[1]:.3f})  err={err:.1f} mm",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    frames.append(out)

renderer.close()
imageio.mimsave(os.path.join(OUT, "opencv_demo.gif"), frames, fps=2)
imageio.imwrite(os.path.join(OUT, "opencv_overlay_example.png"), frames[0])
print(f"{len(frames)} frame salvati in {OUT}")