import numpy as np
from rl.arduinobot_reach_env import ArduinobotReachEnv
from rl.ik_baseline import DLSController

# Set the environment, the controller and the parameters for testing
env = ArduinobotReachEnv()
ctrl = DLSController(env)
N, ok, steps, dist = 50, 0, [], []
env.reset(seed=0)

for _ in range(N):

    # Initialize the episode and set the counter steps to 0
    done, n = False, 0
    
    # Controller run the robot
    while not done:
        # The results ignores observation and reward (first two outputs)
        _, _, t, tr, info = env.step(ctrl.action())
        n += 1
        done = t or tr
    
    # Store the results of the episode
    ok += int(info["is_success"])
    steps.append(n)
    dist.append(info["distance"])
    env.reset()

print(f"IK oracle: success {ok}/{N}, mean steps {np.mean(steps):.1f}, "
      f"Final mean steps {1000 * np.mean(dist):.1f} mm")