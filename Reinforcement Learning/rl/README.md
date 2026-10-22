# RL Module — PPO Reaching in MuJoCo Driven by Vision

## Overview

This module implements a Proximal Policy Optimization (PPO) controller using Stable-Baselines3 to move the end effector of the Arduinobot robotic arm (joints 1–3) toward a 3D target in MuJoCo.

The project combines three components:

- **Reinforcement learning:** a PPO policy trained to reach target positions.
- **Classical control baseline:** damped least-squares inverse kinematics (DLS).
- **Computer vision:** OpenCV-based cube detection and pixel-to-world projection, described in `vision/README.md`.

The intended pipeline is:

1. Detect the cube in the camera image.
2. Estimate its 3D position in the robot's world coordinate frame.
3. Pass the target to the controller.
4. Move the end effector toward the target and evaluate the final distance.

The PPO controller and vision pipeline have been evaluated separately. Full end-to-end reaching from the vision-estimated target has not yet been demonstrated successfully.

## Environment

- **Observation space (15 values):** joint angles, joint velocities, end-effector position, target position, and target-minus-end-effector displacement.
- **Action space (3 values):** joint-position setpoint increments, limited to 0.08 rad per step.
- **Control frequency:** 20 Hz.
- **Reward:** negative end-effector-to-target distance, with an action penalty of `0.01 × |action|²`; a reward of +5 is given on success.
- **Success condition:** distance to target below 0.03 m (3 cm).
- **Episode length:** up to 100 steps (5 seconds).
- **Target sampling:** target positions are sampled from a configured box within the selected workspace region.

The training target bounds are:

| Axis | Minimum (m) | Maximum (m) |
|---|---:|---:|
| X | -0.25 | 0.25 |
| Y | 0.15 | 0.45 |
| Z | 0.30 | 0.50 |

## Simplifications

The following assumptions are deliberate and apply to the simulation:

- Position actuators control joints 1–3.
- The gripper is not actuated; grasping is not implemented.
- Gravity compensation is applied to all links.
- Self-collisions are disabled; floor contact is retained.
- Targets are restricted to a selected region rather than the entire theoretical workspace.
- The trained policy is evaluated within the target distribution for which it was developed.

## Training

The PPO controller uses Stable-Baselines3 with an MLP policy network containing two hidden layers of 128 units each.

Training configuration:

- Algorithm: PPO
- Policy: `MlpPolicy`
- Hidden layers: 128 × 128
- Training budget: 2,000,000 timesteps
- Parallel environments: 4
- Device: CPU
- Seed used for the documented training run: 42
- Model: `rl_models/ppo_v2_corrected_final.zip`
- Best checkpoint: `rl_models/ppo_v2_corrected_best/best_model.zip`

Training curves are saved in `docs/rl_training_curves.png`.

## Results: Controller Comparison

The following results were obtained in a 200-episode evaluation using the same target set for all three controllers.

| Policy | Success rate | Mean time to success | Mean final error | End-effector path |
|---|---:|---:|---:|---:|
| Random actions | 0.0% | N/A | 1226.3 mm | 3.26 m |
| Damped least-squares IK | 89.0% | 1.96 s | 30.4 mm | 1.37 m |
| PPO | 100.0% | 1.36 s | 25.8 mm | 1.70 m |

### Interpretation

PPO achieved the highest success rate and the lowest mean final error in this evaluation. It also reached successful targets faster than the DLS baseline, although its mean end-effector path was longer.

The random-action baseline did not solve the task.

Additional PPO evaluations produced the following results:

- **100 episodes:** 100/100 successes (100.0%), mean final distance 25.1 mm.
- **1,000 episodes:** 999/1,000 successes (99.9%), mean final distance 25.1 mm and mean episode length of 25.76 steps.

These additional evaluations support the conclusion that the trained policy performs consistently well within its training target distribution. They do not establish equivalent performance for targets outside that distribution.

## Vision-to-Policy Integration

The vision module detects a colored cube in the rendered camera image and estimates its 3D position using pixel-to-world projection.

### Perception results

For a representative test with the cube on the plane `z = 0.04 m`:

- Cube detected: 1/1 trial.
- True cube position: approximately `(0.532, 0.704, 0.040) m`.
- Estimated position: approximately `(0.535, 0.706, 0.040) m`.
- Position estimation error: **4.1 mm**.

This result shows that the projection can estimate the cube position accurately under the tested camera and plane configuration.

### End-to-end reaching results

In the same test, the estimated target was passed to the PPO policy. The target was outside the policy's training bounds, particularly along X and Y.

| Metric | Result |
|---|---:|
| Cubes detected | 1/1 |
| Mean perception error | 4.1 mm |
| Final distance to the cube | 185.7 mm |
| Success threshold | 30 mm |
| End-to-end success | 0/1 (0%) |

Although perception was accurate, the controller did not reach the cube. This is consistent with the target being outside the distribution used to train the PPO policy.

### Isolated PPO diagnostic

A separate diagnostic test used a target within the PPO training bounds and assigned the true target directly to the environment, bypassing the vision estimate.

- Final distance to target: 27.5 mm.
- Success threshold: 30 mm.
- Episode length: 23 steps.
- PPO reaching result: successful.

In that diagnostic, the cube was placed at approximately `z = 0.40 m`, while the vision projection still assumed `z = 0.04 m`. The reported perception error of approximately 375 mm therefore reflects an inconsistent projection plane. It is not a valid measure of perception accuracy at the elevated cube position.

**Important:** the isolated PPO diagnostic is not an end-to-end vision-to-policy success because the controller received the true target instead of the vision-estimated position.

A 30-trial end-to-end evaluation has not yet been completed. Therefore, no 30-trial detection rate, mean perception error, or end-to-end success rate is claimed here.

## Automated Tests

The repository test suite was run with:

```bash
python3 -m pytest -v
```

Result:

**8 passed in 2.01 seconds.**

The eight collected tests cover:

- RL environment observation and step shapes.
- Successful termination when the target is at the end effector.
- Episode truncation when the target remains far away.
- Target-pool bounds.
- Vision position error in three test cases.
- Rejection of a cube outside the camera's field of view.

These tests confirm that all currently collected automated tests pass. They do not test successful end-to-end reaching using the vision-estimated target.

## Limitations

- Simulation only; no sim-to-real validation.
- No physical robot validation.
- Single colored cube and fixed camera configuration.
- The projection assumes a known cube plane height.
- Only three robot joints are controlled.
- Gripper control and grasping are not implemented.
- PPO performance is established for its training target distribution, not arbitrary workspace positions.
- The complete vision-to-policy reaching pipeline remains to be validated with a consistent target geometry and controller workspace.

## Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Analyze the workspace:

```bash
python3 -m rl.workspace_analysis
```

Train PPO:

```bash
python3 -m rl.train_ppo --steps 2000000 --envs 4 --name ppo_v2_corrected --seed 42
```

Evaluate the controllers:

```bash
python3 -m rl.evaluate
```

Run the vision-to-policy simulation:

```bash
python3 -m rl.vision_to_policy --trials 30 --seed 5
```

Run the automated tests:

```bash
python3 -m pytest -v
```

## Demo
![RL reaching demo](../docs/rl_demo.gif)
