import argparse
import os

import cv2
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from rl import config as C
from rl.arduinobot_reach_env import ArduinobotReachEnv
from rl.ik_baseline import DLSController


def label(frame, title, info):
    f = frame.copy()

    # Overlay the controller name, current target distance, and success status on the frame.
    txt = f"{title} | err {1000 * info['distance']:.0f} mm" + (
        "  SUCCESS" if info["is_success"] else ""
    )
    cv2.putText(f, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(C.MODELS_DIR, "ppo_v2_corrected_best", "best_model.zip"))
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    # Create separate rendering environments to compare the IK baseline and PPO policy side by side.
    env_ik = ArduinobotReachEnv(render_mode="rgb_array")
    env_rl = ArduinobotReachEnv(render_mode="rgb_array")

    ik = DLSController(env_ik)
    model = PPO.load(args.model, device="cpu")

    frames = []

    env_ik.reset(seed=args.seed)

    # Use the same seed for both environments to initialize comparable target scenarios.
    obs, _ = env_rl.reset(seed=args.seed)

    for _ in range(args.episodes):
        done_i = done_r = False
        info_i = info_r = {"distance": 0.0, "is_success": False}

        # Advance both controllers independently until both episodes have terminated.
        while not (done_i and done_r):

            # Step the IK environment only while its episode is still active.
            if not done_i:
                _, _, t, tr, info_i = env_ik.step(ik.action())
                done_i = t or tr

            # Run the PPO policy deterministically and step its environment.
            if not done_r:
                a, _ = model.predict(obs, deterministic=True)
                obs, _, t, tr, info_r = env_rl.step(a)
                done_r = t or tr

            # Render both environments, annotate their frames, and combine them horizontally.
            frames.append(
                np.hstack([
                    label(env_ik.render(), "IK (baseline)", info_i),
                    label(env_rl.render(), "PPO policy", info_r)
                ])
            )

        # Hold the final frame briefly to make the end of each episode easier to observe.
        frames.extend([frames[-1]] * 10)

        # Reset both environments before starting the next comparison episode.
        env_ik.reset()
        obs, _ = env_rl.reset()

    os.makedirs(C.DOCS_DIR, exist_ok=True)

    # Export the complete comparison as an MP4 video and a lower-frame-rate GIF.
    imageio.mimsave(os.path.join(C.DOCS_DIR, "rl_demo.mp4"), frames, fps=20, macro_block_size=1)
    imageio.mimsave(os.path.join(C.DOCS_DIR, "rl_demo.gif"), frames[::2], fps=10)
    print("Saved docs/rl_demo.mp4 and docs/rl_demo.gif")


if __name__ == "__main__":
    main()