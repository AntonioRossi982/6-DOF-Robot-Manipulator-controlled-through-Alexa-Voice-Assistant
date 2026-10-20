
import cv2
import numpy as np
from vision import config

# Initial range of magenta in HSV.
# OpenCV uses H in the range 0-179.
# S and V are in the range 0-255.
LOWER = np.array([135, 100, 50]) # H min = 135, S min = 100, V min = 50
UPPER = np.array([165, 255, 255]) # H max = 165, S max = 255, V max = 255

MIN_AREA = config.MIN_AREA  # Contour minimum area in pixels^2


def segment_cube(frame_rgb):
    """Returns a binary mask of the magenta pixels."""
    # this conversion is necessary because OpenCV uses BGR instead of RGB
    # HSV is useful for color segmentation because it separates the color 
    # information (hue) from the intensity information (saturation and value)
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV) 

    # The mask has upper and lower bounds defined previously
    mask = cv2.inRange(hsv, LOWER, UPPER)

    # Define a kernel for morphological operations
    # 5x5 is a common choice but you can manipulate it based on your necessity.
    kernel = np.ones((config.KERNEL_SIZE,) * 2, np.uint8)

    # Remove small isolated elements in the detected region.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    # Close small holes in the detected region.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    return mask


def detect_cube(frame_rgb):
    """Returns the center, area, and contour of the cube, or None."""
    # Apply the segmentation to get the binary mask of the magenta pixels in the frame.
    mask = segment_cube(frame_rgb)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL, # consider external contours only
        cv2.CHAIN_APPROX_SIMPLE # Compress consecutive points
    )

    if not contours:
        return None

    # Select the contour with the largest area.
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)

    # If the area is smaller than the minimum area, then we consider
    # that the cube is not detected and return None.
    if area < MIN_AREA:
        return None

    # Calculate the moments of the contour to find its center.
    # u = m10/m00 and v = m01/m00 (based on area and centroid)
    moments = cv2.moments(best)

    # If the moment of 0 order is zero, it means that the contour has no area and we cannot compute its center.
    if moments["m00"] == 0:
        return None

    # Calculate the center of the contour using the moments.
    u = moments["m10"] / moments["m00"]
    v = moments["m01"] / moments["m00"]

    return {
        "u": u,
        "v": v,
        "area": area,
        "contour": best
    }
