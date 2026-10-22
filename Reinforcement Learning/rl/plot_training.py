
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl import config as C

NAME = "ppo_v2_corrected"

# Load evaluation data recorded during PPO training.
d = np.load(os.path.join(C.RUNS_DIR, NAME + "_eval", "evaluations.npz"))
t = d["timesteps"]

# Create two subplots to visualize training performance.
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

# Plot the mean evaluation reward at each training timestep.
# Higher rewards generally indicate better policy performance.
ax[0].plot(t, d["results"].mean(axis=1))
ax[0].set_xlabel("Timesteps")
ax[0].set_ylabel("Mean evaluation reward")
ax[0].grid(alpha=0.3)

# Plot the success rate if success data is available in the evaluation file.
# Convert the mean success fraction to a percentage.
if "successes" in d.files:
    ax[1].plot(t, 100 * d["successes"].mean(axis=1))
    ax[1].set_xlabel("Timesteps")
    ax[1].set_ylabel("Success rate [%]")
    ax[1].grid(alpha=0.3)
# Adjust the layout to prevent labels from overlapping.
plt.tight_layout()

# Save the training curves as an image for later analysis.
os.makedirs(C.DOCS_DIR, exist_ok=True)
plt.savefig(os.path.join(C.DOCS_DIR, "rl_training_curves.png"), dpi=150)

print("Saved docs/rl_training_curves.png")