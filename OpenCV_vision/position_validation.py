
import os
import numpy as np
import imageio.v2 as imageio

from vision import config
from vision.scene import load_scene, place_cube, render
from vision.cube_detector import detect_cube
from vision.projection import pixel_to_world


OUT = os.path.expanduser("~/mujoco_robotics/data/frames")
N_TRIALS = 20

os.makedirs(OUT, exist_ok=True)

model, data, renderer, cube_id = load_scene()
rng = np.random.default_rng(0)

errors = []
failures = 0

try:
    for i in range(N_TRIALS):
        x = rng.uniform(0.25, 0.60)
        y = rng.uniform(0.30, 0.80)

        true = place_cube(model, data, cube_id, x, y)
        frame = render(renderer, data)
        det = detect_cube(frame)

        if det is None:
            failures += 1
            imageio.imwrite(
                os.path.join(OUT, f"fail_{i:02d}.png"),
                frame
            )
            print(f"{i:02d} - Cube not detected")
            continue

        est = pixel_to_world(
            model, data, config.CAMERA,
            det["u"], det["v"],
            config.IMG_W, config.IMG_H,
            config.CUBE_Z
        )

        err_mm = np.linalg.norm(est[:2] - true[:2]) * 1000
        errors.append(err_mm)

        print(
            f"{i:02d} True=({true[0]:.3f},{true[1]:.3f}) "
            f"Estimated=({est[0]:.3f},{est[1]:.3f}) "
            f"Error={err_mm:.1f} mm"
        )

    if errors:
        e = np.array(errors)
        print(
            f"\nDetected {len(e)}/{N_TRIALS} "
            f"| missed {failures} "
            f"| average error {e.mean():.1f} mm "
            f"| max {e.max():.1f} mm "
            f"| std {e.std():.1f} mm"
        )
    else:
        print(f"\nNo detections in {N_TRIALS} trials.")

    # Negative test
    place_cube(model, data, cube_id, 5.0, 5.0)
    negative_frame = render(renderer, data)
    negative_det = detect_cube(negative_frame)

    if negative_det is None:
        print("Negative test: PASS (no detection)")
    else:
        print(
            "Negative test: FAIL (false positive; "
            f"area={negative_det['area']:.1f} px^2)"
        )

finally:
    renderer.close()
