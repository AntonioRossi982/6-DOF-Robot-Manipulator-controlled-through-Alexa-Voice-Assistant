import argparse
import csv
import os

# Configure MuJoCo to use EGL for headless rendering when no backend is specified.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import imageio.v2 as imageio
from stable_baselines3 import PPO

from rl import config as C
from rl.arduinobot_reach_env import ArduinobotReachEnv
from vision import config as V
from vision.cube_detector import detect_cube
from vision.projection import pixel_to_world


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(C.MODELS_DIR, "ppo_v2_corrected_best", "best_model.zip"),)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--video_eps", type=int, default=4)
    args = p.parse_args()

    # Initialize the simulation, load the trained PPO policy, and configure the camera renderer.
    env = ArduinobotReachEnv(model_path=C.RL_VISION_MODEL, render_mode="rgb_array")
    policy = PPO.load(args.model, device="cpu")
    cam = mujoco.Renderer(env.model, V.IMG_H, V.IMG_W)
    cube_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, V.CUBE_BODY)
    rng = np.random.default_rng(args.seed)

    rows, frames = [], []
    for k in range(args.trials):

        # Randomize the cube position and reset the environment for a new trial.
        # true = np.array([rng.uniform(*V.SPAWN_X), rng.uniform(*V.SPAWN_Y), V.CUBE_Z])
        true = np.array([rng.uniform(*C.TARGET_X), rng.uniform(*C.TARGET_Y), rng.uniform(*C.TARGET_Z)])
        env.reset(seed=int(rng.integers(1_000_000)), options={"target": true, "move_markers": False})
        env.model.body_pos[cube_id] = true
        mujoco.mj_forward(env.model, env.data)

        # 1) Perception: capture an image with the robot in its initial configuration.
        cam.update_scene(env.data, camera=V.CAMERA)
        det = detect_cube(cam.render())

        if det is None:
            # Record a failed detection and skip the control stage for this trial.
            rows.append({"trial": k, "detected": False, "perception_err_mm": float("nan"),
                         "final_dist_to_cube_mm": float("nan"), "success": False})
            continue

        # Convert the detected pixel coordinates into an estimated 3D world position.
        estimation = pixel_to_world(env.model, env.data, V.CAMERA, det["u"], det["v"],
                             V.IMG_W, V.IMG_H, V.CUBE_Z)

        print("Target reale:", true)
        print("Target stimato:", estimation)
        print("Errore di proiezione [mm]:", 1000 * np.linalg.norm(estimation - true))
        print("Regione training X:", C.TARGET_X)
        print("Regione training Y:", C.TARGET_Y)
        print("Regione training Z:", C.TARGET_Z)

        # 2) Control: use the estimated cube position as the policy's target.
        env.target = estimation.copy()
        # env.target = true.copy()
        obs = env._obs()

        print("Target reale:", true)
        print("Target policy:", env.target)
        print("EE iniziale:", env.data.xpos[env.ix["ee"]])
        print("Errore iniziale:", np.linalg.norm(env.target - env.data.xpos[env.ix["ee"]]))
        print("Ultimi 9 valori osservazione:", obs[-9:])

        done = False
        # while not done:
        #     action, _ = policy.predict(obs, deterministic=True)
        #     obs, _, term, trunc, _ = env.step(action)
        #     done = term or trunc
        step_debug = 0

        while not done:
            action, _ = policy.predict(obs, deterministic=True)

            if step_debug < 10:
                ee_before = env.data.xpos[env.ix["ee"]].copy()
                dist_before = np.linalg.norm(env.target - ee_before)

            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            if step_debug < 10:
                ee_after = env.data.xpos[env.ix["ee"]].copy()
                dist_after = np.linalg.norm(env.target - ee_after)

                print(
                    f"Step {step_debug + 1}: "
                    f"action={np.round(action, 3)}, "
                    f"dist={dist_before:.3f}->{dist_after:.3f} m, "
                    f"EE={np.round(ee_after, 3)}, "
                    f"reward={reward:.3f}"
                )

            step_debug += 1

            # Save rendered frames only for the first requested video trials.
            if k < args.video_eps:
                frames.append(env.render())

        print("Passi eseguiti:", step_debug)
        print("Distanza finale:", np.linalg.norm(
            env.target - env.data.xpos[env.ix["ee"]]
        ))
        print("Target:", env.target)
        print("EE finale:", env.data.xpos[env.ix["ee"]])
        print("Soglia successo:", C.SUCCESS_RADIUS)

        # 3) Evaluation: measure the final distance to the true cube position,
        # rather than to the estimated target used by the policy.
        final = float(np.linalg.norm(env.data.xpos[env.ix["ee"]] - true))
        rows.append({"trial": k, "detected": True,
                     "perception_err_mm": 1000 * float(np.linalg.norm(estimation - true)),
                     "final_dist_to_cube_mm": 1000 * final,
                     "success": final < C.SUCCESS_RADIUS})

    cam.close()

    # Summarize perception and end-to-end performance.
    det_rows = [r for r in rows if r["detected"]]
    print(f"Detected cubes: {len(det_rows)}/{args.trials}")

    if det_rows:
        print(f"Mean perception error: {np.mean([r['perception_err_mm'] for r in det_rows]):.1f} mm")
        print(f"End-to-end success rate (all {args.trials} trials): "
              f"{100 * np.mean([r['success'] for r in rows]):.1f}%")
        print(f"Mean final distance to the cube: "
              f"{np.mean([r['final_dist_to_cube_mm'] for r in det_rows]):.1f} mm")

    # Export the per-trial results to a CSV file.
    os.makedirs(C.DOCS_DIR, exist_ok=True)
    with open(os.path.join(C.DOCS_DIR, "rl_vision_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Save a video if at least one trial was selected for recording.
    if frames:
        imageio.mimsave(os.path.join(C.DOCS_DIR, "vision_to_policy.mp4"), frames, fps=20,
                        macro_block_size=1)
        print("Saved docs/vision_to_policy.mp4")


if __name__ == "__main__":
    main()