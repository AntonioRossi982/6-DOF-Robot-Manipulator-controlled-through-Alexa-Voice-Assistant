import time
import numpy as np
from stable_baselines3.common.env_checker import check_env

from rl.arduinobot_reach_env import ArduinobotReachEnv

# Check the environment using stable_baselines3's check_env function
env = ArduinobotReachEnv()
check_env(env, warn=True)
print("check_env is correctly settled up")

# Run a simple test of the environment with random actions
t0 = time.time()
steps, ep, successes, dists = 0, 0, 0, []
env.reset(seed=0)

while ep < 20:
    
    # Generate a random action and update step in the environment (for each call of env.step())
    # Note: observtion and reward are not relevant for this test
    _, _, term, trunc, info = env.step(env.action_space.sample())
    steps += 1

    # Stop the episode if the robot has reached the target or if the maximum number of steps is reached
    if term or trunc:
        ep += 1
        successes += int(info["is_success"])
        dists.append(info["distance"])
        # reset the environment for the next episode
        env.reset()

dt = time.time() - t0 # evaluate test time
print(f"{steps / dt:.0f} step/s | Random policy: success {successes}/20, "
      f"final distance average {np.mean(dists):.3f} m")