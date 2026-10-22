import argparse
import csv
import os

import numpy as np
from stable_baselines3 import PPO

from rl import config as C
from rl.arduinobot_reach_env import ArduinobotReachEnv
from rl.ik_baseline import DLSController


def run_episodes(env, policy, n_episodes, seed):

    results = []
    obs, _ = env.reset(seed=seed)

    for _ in range(n_episodes):

        done, steps, path = False, 0, 0.0
        prev = env.data.xpos[env.ix["ee"]].copy()

        while not done:

            # Apply the policy action and advance the environment by one control step.
            obs, _, term, trunc, info = env.step(policy(obs))

            # Accumulate the end-effector displacement to estimate the total Cartesian path length.
            ee = env.data.xpos[env.ix["ee"]]
            path += float(np.linalg.norm(ee - prev))
            prev = ee.copy()

            steps += 1
            # An episode ends either when a terminal condition is reached or when the time limit expires.
            done = term or trunc

        results.append({
            "success": bool(info["is_success"]),
            "steps": steps,
            "final_dist": info["distance"],
            "path": path
        })

        # Reset the environment before starting the next episode.
        obs, _ = env.reset()

    return results


def summarize(name, res, dt_ctrl):

    succ = np.array([r["success"] for r in res])
    steps = np.array([r["steps"] for r in res])

    return {
        "policy": name,

        # Compute the fraction of episodes in which the robot successfully reaches the target.
        "success_rate": float(succ.mean()),
        
        # Estimate the mean time to success using only successful episodes.
        # If no episode succeeds, return NaN because the metric is undefined.
        "time_to_success_s": float(steps[succ].mean() * dt_ctrl) if succ.any() else float("nan"),

        # Convert the mean final distance from meters to millimeters.
        "final_error_mm": 1000 * float(np.mean([r["final_dist"] for r in res])),

        # Compute the mean total Cartesian distance traveled by the end-effector.
        "ee_path_m": float(np.mean([r["path"] for r in res])),
    }


def main():

    # Define command-line arguments for the evaluation configuration.
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(C.MODELS_DIR, "ppo_v2_corrected_best", "best_model.zip"))
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    # Initialize the list used to collect metrics for each policy.
    rows = []

    # Evaluate random actions as a baseline to establish reference performance.
    env = ArduinobotReachEnv()

    # Compute the duration of one control step from the simulation timestep and number of substeps.
    dt_ctrl = C.N_SUBSTEPS * env.model.opt.timestep
    env.action_space.seed(args.seed)

    rows.append(summarize("random actions", run_episodes(env, lambda o: env.action_space.sample(),
            args.episodes, args.seed), dt_ctrl))

    # Evaluate the Damped Least Squares (DLS) inverse kinematics controller.
    # This provides a conventional model-based baseline for comparison with the learned policy.
    env = ArduinobotReachEnv()
    ik = DLSController(env)

    rows.append(summarize("Inverse kinematics (DLS)", run_episodes(env, lambda o: ik.action(),
            args.episodes, args.seed), dt_ctrl))

    # Load the trained PPO model and evaluate its policy in a separate environment.
    env = ArduinobotReachEnv()
    model = PPO.load(args.model, device="cpu")

    rows.append(summarize("Proximal Policy Optimization (PPO)", run_episodes(env,
        lambda o: model.predict(o, deterministic=True)[0], args.episodes, args.seed), dt_ctrl))

    # Print the evaluation metrics side by side to compare the three control strategies.
    print(f"\n{'policy':10s} {'success':>8s} {'time [s]':>9s} {'error [mm]':>11s} {'EE path [m]':>12s}")
    for r in rows:
        print(f"{r['policy']:10s} {100 * r['success_rate']:7.1f}% {r['time_to_success_s']:9.2f} "
              f"{r['final_error_mm']:11.1f} {r['ee_path_m']:12.2f}")

    os.makedirs(C.DOCS_DIR, exist_ok=True)
    with open(os.path.join(C.DOCS_DIR, "rl_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()