
import numpy as np
import mujoco

from rl import config as C
from rl.model_utils import load_model, indices

model = load_model()
data = mujoco.MjData(model)
ix = indices(model)

rng = np.random.default_rng(0)
pts = np.empty((20_000, 3))

for i in range(len(pts)):
    data.qpos[ix["qadr"]] = rng.uniform(ix["low"], ix["high"])
    mujoco.mj_forward(model, data)
    pts[i] = data.xpos[ix["ee"]]

print("Minimum :", pts.min(axis=0))
print("Maximum :", pts.max(axis=0))
print("First sample:", pts[0])

print("\nTarget configured:")
print("X:", C.TARGET_X)
print("Y:", C.TARGET_Y)
print("Z:", C.TARGET_Z)

mask = (
    (pts[:, 0] >= C.TARGET_X[0]) &
    (pts[:, 0] <= C.TARGET_X[1]) &
    (pts[:, 1] >= C.TARGET_Y[0]) &
    (pts[:, 1] <= C.TARGET_Y[1]) &
    (pts[:, 2] >= C.TARGET_Z[0]) &
    (pts[:, 2] <= C.TARGET_Z[1])
)

print(f"\nSamples in target: {mask.sum()}/{len(pts)}")
