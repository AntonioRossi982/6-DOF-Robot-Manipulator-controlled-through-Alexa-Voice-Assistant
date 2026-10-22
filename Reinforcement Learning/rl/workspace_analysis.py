import numpy as np
import mujoco

from rl import config as C
from rl.model_utils import load_model, indices
from vision import config as V

N = 200_000
CELL = 0.05      # length of the cells [m]
SLAM = 0.03      # half-width of the plane considered [m]
MIN_PTS = 5      # minimum number of points in a cell to consider it reachable
MARGIN = 0.08    # margin inside the camera view [m]
Z_MIN = 0.02     # below this height is the floor

model = load_model()
data = mujoco.MjData(model)
ix = indices(model)
rng = np.random.default_rng(0)

# Monte Carlo sampling of the reachable workspace of the end-effector
pts = np.empty((N, 3))
for i in range(N):
    data.qpos[ix["qadr"]] = rng.uniform(ix["low"], ix["high"])
    mujoco.mj_kinematics(model, data)
    pts[i] = data.xpos[ix["ee"]]
pts = pts[pts[:, 2] > Z_MIN] # select only points above the floor (boolean mask)

print(f"Samples above the floor: {len(pts)}")
print("Bounding box reachable (x, y, z):",
      [(round(float(pts[:, k].min()), 2), round(float(pts[:, k].max()), 2)) for k in range(3)])

# Divide the workspace into square 3D cells of length 5 cm
xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
nx = int(np.ceil((xmax - xmin) / CELL)) + 1
ny = int(np.ceil((ymax - ymin) / CELL)) + 1
xc = xmin + (np.arange(nx) + 0.5) * CELL
yc = ymin + (np.arange(ny) + 0.5) * CELL


def occupancy(zc):
    
    """ Occupancy grid of the reachable workspace at height zc.
    This function aim to build a reachable workspace map on the horizontal plane."""

    sel = pts[np.abs(pts[:, 2] - zc) < SLAM]
    counts = np.zeros((nx, ny), dtype=int)
    
    if len(sel):
        ii = ((sel[:, 0] - xmin) // CELL).astype(int)
        jj = ((sel[:, 1] - ymin) // CELL).astype(int)
        np.add.at(counts, (ii, jj), 1)
    
    return counts >= MIN_PTS


def in_view(zc):
    
    """Function to check if the cells are visible from the camera at height zc."""

    d = C.CAM_POS[2] - zc
    
    if d <= 0:
        return np.zeros((nx, ny), dtype=bool)
    
    half_h = d * np.tan(np.deg2rad(C.CAM_FOVY) / 2)
    half_w = half_h * C.IMG_W / C.IMG_H
    x0, x1 = C.CAM_POS[0] - half_w + MARGIN, C.CAM_POS[0] + half_w - MARGIN
    y0, y1 = C.CAM_POS[1] - half_h + MARGIN, C.CAM_POS[1] + half_h - MARGIN

    return ((xc[:, None] >= x0) & (xc[:, None] <= x1) &
            (yc[None, :] >= y0) & (yc[None, :] <= y1))


def largest_rectangle(grid):

    """Rectangle of True cells with maximum area (histogram method)."""

    n, m = grid.shape
    heights = np.zeros(m, dtype=int)
    best = (0, None)
    
    for r in range(n):
        heights = np.where(grid[r], heights + 1, 0)
        stack = []
        
        for c in range(m + 1):
            h = heights[c] if c < m else 0
            start = c
            
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                area = sh * (c - s)
                if area > best[0]:
                    best = (area, (r - sh + 1, r, s, c - 1))
                start = s
            stack.append((start, h))
    return best


rows = []

for zc in np.arange(0.10, 1.31, 0.05):
    area_cells, rect = largest_rectangle(occupancy(zc) & in_view(zc))
    
    if rect is None:
        continue
    
    r0, r1, c0, c1 = rect
    box = (xmin + r0 * CELL, xmin + (r1 + 1) * CELL, ymin + c0 * CELL, ymin + (c1 + 1) * CELL)
    rows.append((float(zc), area_cells * CELL ** 2, box))
rows.sort(key=lambda t: -t[1])

print("\nBest planes for the cube (full rectangle, reachable and visible from the camera):")
for zc, area, (x0, x1, y0, y1) in rows[:6]:
    print(f"  z={zc:.2f}  area={area:.2f} m2  x[{x0:.2f},{x1:.2f}]  y[{y0:.2f},{y1:.2f}]")

# current setup of the vision module
x0, x1, y0, y1 = 0.25, 0.60, 0.30, 0.80
mask = ((xc[:, None] >= x0) & (xc[:, None] <= x1) & (yc[None, :] >= y0) & (yc[None, :] <= y1))
cov = (
    (occupancy(V.CUBE_Z) & in_view(V.CUBE_Z))[mask].mean()
    if mask.any()
    else 0.0
)

print(f"\nCurrent setup (z={V.CUBE_Z}, box x[{x0},{x1}] y[{y0},{y1}]): "
      f"{100 * cov:.0f}% of the reachable cells")