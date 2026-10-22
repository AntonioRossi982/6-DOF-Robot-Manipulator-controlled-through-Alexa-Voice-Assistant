import numpy as np
import pytest

from rl import config as C
from rl.arduinobot_reach_env import ArduinobotReachEnv


# Create one environment for all tests in this module and close it afterward.
@pytest.fixture(scope="module")
def env():
    e = ArduinobotReachEnv()
    yield e
    e.close()


def test_reset_and_step_shapes(env):
    # Verify that reset() returns an observation with the expected shape.
    obs, _ = env.reset(seed=0)
    assert obs.shape == (15,)

    # Execute one step with a zero action and check the observation and reward.
    obs, r, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
    assert obs.shape == (15,) and np.isfinite(r)


def test_success_when_target_at_end_effector(env):
    env.reset(seed=0)

    # Set the target to the end-effector's current position.
    ee = env.data.xpos[env.ix["ee"]].copy()
    env.reset(seed=0, options={"target": ee})

    # Verify that the environment reports success when the target is reached.
    _, _, term, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert term and info["is_success"]


def test_truncation_when_target_far(env):
    env.reset(seed=0)

    # Place the target far from the end-effector to prevent successful completion.
    ee = env.data.xpos[env.ix["ee"]].copy()
    env.reset(seed=0, options={"target": ee + np.array([0.6, 0.0, 0.0])})

    # Run the maximum number of steps and verify that the episode is truncated.
    for _ in range(C.MAX_STEPS):
        _, _, term, trunc, _ = env.step(np.zeros(3, dtype=np.float32))
    assert trunc and not term


def test_target_pool_inside_configured_box(env):
    # Check that every target in the pool lies within the configured XYZ bounds.
    p = env.pool
    assert (p[:, 0] >= C.TARGET_X[0]).all() and (p[:, 0] <= C.TARGET_X[1]).all()
    assert (p[:, 1] >= C.TARGET_Y[0]).all() and (p[:, 1] <= C.TARGET_Y[1]).all()
    assert (p[:, 2] >= C.TARGET_Z[0]).all() and (p[:, 2] <= C.TARGET_Z[1]).all()