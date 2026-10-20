
import os
import sys
# Import the parent directory to access the vision module.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..")
)

import cv2
import numpy as np
import imageio.v2 as imageio
from vision.cube_detector import detect_cube, segment_cube
# Define the path to the data directory where the images are stored.
DATA = os.path.expanduser("~/mujoco_robotics/data/frames")
# Read the PNG image from the data directory and keep only the first 3 channels (RGB)
frame = imageio.imread(
    os.path.join(DATA, "frame_000.png")
)[:, :, :3]

# Convert the RGB image to HSV to analyze the hue distribution.
hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

sat = hsv[:, :, 1] > 100 # Consider only the pixel with saturation greater than 100.
# Extract the Hues from selected pixels and count the number of times they appear in the image. 
hues, counts = np.unique(
    hsv[:, :, 0][sat],
    return_counts=True
)
# Order the Hues (pixels) by their frequency and select the top 8 most frequent.
top = np.argsort(counts)[::-1][:8]

print(
    "Hue most frequent (hue, pixel):",
    [(int(hues[i]), int(counts[i])) for i in top]
)

# Segmentation and detection of the cube in the image.
mask = segment_cube(frame) # binary mask
det = detect_cube(frame) # detection result (find the bigger contour and evaluate the centroid)

print(
    "Detection:",
    None if det is None else (
        round(det["u"], 1),
        round(det["v"], 1),
        det["area"]
    )
)

# Draw the detection results on a copy of the original RGB frame.
overlay = frame.copy()
# If the cube is detected, draw the contour and the centroid
if det is not None:
    cv2.drawContours(
        overlay, [det["contour"]], -1, (0, 255, 0), 1
    ) # Image on which to draw, contours, contour index (-1 for all), color (green), thickness

    cv2.circle(
        overlay,
        (int(round(det["u"])), int(round(det["v"]))),
        2,
        (255, 255, 255),
        -1
    ) # Image on which to draw, center coordinates, radius, color (white), thickness (-1 for filled circle)
    
    # Check the detection results and print the center and the area
    print(f"Cube Center: u={det['u']:.2f}, v={det['v']:.2f}")
    print(f"Area of the contour: {det['area']:.1f} pixel²")
else:
    print("No cube detected.")
# Save the image (mask and overlay) in the image directory
imageio.imwrite(
    os.path.join(DATA, "mask_000.png"), mask
)

imageio.imwrite(
    os.path.join(DATA, "overlay_000.png"), overlay
)

print("Image saved in", DATA)