# MuJoCo–OpenCV Cube Detection and Position Estimation

## Overview

This project implements and validates a computer vision pipeline for detecting a colored cube in a simulated robotic environment. The simulation is built with **MuJoCo**, while **OpenCV** processes rendered camera images to identify the cube and estimate its position in world coordinates.

The project was developed incrementally over four days, progressing from scene creation and image rendering to color-based object detection, geometric position estimation, configurable components, automated tests, and visual demonstrations.

The main objective is to demonstrate how a simulated camera image can be used to detect an object and estimate its planar position in the robot's world coordinate system.

### Key features

- MuJoCo simulation with a dedicated top-down camera.
- A fixed magenta cube used as a visual target.
- OpenCV-based color segmentation in HSV color space.
- Morphological image processing and contour-based object detection.
- Pixel-to-world coordinate conversion using camera geometry.
- Quantitative position validation against the cube's known simulated position.
- Automated regression tests using pytest.
- Reusable configuration and scene-management modules.
- A demonstration GIF and an annotated example image.

## Project Structure

The main files relevant to the vision pipeline are organized as follows:

```text
mujoco_robotics/
├── models/
│   └── arduinobot_vision.xml
├── vision/
│   ├── __init__.py
│   ├── config.py
│   ├── scene.py
│   ├── render_scene.py
│   ├── cube_detector.py
│   ├── run_detection.py
│   ├── projection.py
│   ├── position_validation.py
│   ├── demo.py
│   └── README.md
├── tests/
│   └── test_vision.py
├── docs/
│   ├── opencv_demo.gif
│   └── opencv_overlay_example.png
├── data/
│   └── frames/
├── requirements.txt
└── README.md
```

Generated files and exact file availability may vary depending on the current repository state. The `data/frames/` directory stores generated images locally and should not be committed if it is excluded by `.gitignore`.

## System Architecture

The pipeline consists of five main stages:

1. **Scene setup:** MuJoCo loads the XML model, initializes the simulation data, and creates a renderer.
2. **Image acquisition:** The top-down camera renders the simulated scene into an RGB image.
3. **Object detection:** OpenCV segments the magenta cube using HSV thresholds and extracts its image-space centroid.
4. **Position estimation:** The detected pixel coordinates are projected onto a known horizontal plane in the world coordinate system.
5. **Validation:** The estimated position is compared with the cube's ground-truth position obtained directly from the simulation.

This setup separates image processing from simulation management and geometric projection, making the code easier to test and maintain.

## Day 1 — MuJoCo Scene and Camera Rendering

### Objective

Create a reproducible simulated environment containing a visually identifiable object and a camera suitable for image-based detection.

### Implementation

The MuJoCo model is defined in:

`models/arduinobot_vision.xml`

The scene includes:

- The simulated environment and robot model.
- A top-down camera named `top_cam`.
- A fixed magenta cube named `target_cube`.
- A camera configuration that provides a view of the target workspace.

The cube has half-extents of approximately **0.04 m**, corresponding to a nominal side length of **0.08 m**. Its center is positioned at a height of **0.04 m** above the reference plane.

The renderer produces images at a resolution of **640 × 480 pixels**. The resulting frames can be saved to `data/frames/` for inspection and subsequent processing.

The rendering setup uses MuJoCo's rendering API and the EGL backend when configured for headless execution under Ubuntu on WSL2.

### Outcome

Day 1 established the simulation and image-acquisition foundation required by the computer vision pipeline. The rendered images provide a controlled and repeatable input for detection and validation.

## Day 2 — OpenCV Color-Based Cube Detection

### Objective

Detect the magenta cube in the rendered image and determine its image-space center.

### Implementation

The detection algorithm is implemented in:

`vision/cube_detector.py`

The pipeline performs the following operations:

1. Convert the input image from RGB to HSV color space.
2. Apply lower and upper HSV thresholds to isolate the magenta region.
3. Apply morphological opening to reduce small foreground noise.
4. Apply morphological closing to fill small gaps in the segmented region.
5. Extract external contours from the resulting binary mask.
6. Select the largest contour and reject detections whose area is below a minimum threshold.
7. Calculate the centroid of the selected contour using image moments.

The default HSV thresholds are:

- Lower bound: `(135, 100, 50)`
- Upper bound: `(165, 255, 255)`

The minimum accepted contour area is **50 pixels**, and the morphological kernel size is **5 × 5 pixels**.

For a detected contour, the centroid is calculated from its spatial moments:

\[
u_c = \frac{M_{10}}{M_{00}},
\qquad
v_c = \frac{M_{01}}{M_{00}}
\]

where \(M_{00}\) is the contour's zeroth-order moment, and \(M_{10}\) and \(M_{01}\) are the first-order moments.

The detector returns the centroid in pixel coordinates together with detection information, such as the contour area. If no valid contour is found, the function returns `None`.

The script `vision/run_detection.py` integrates the detector with rendered images and produces visual outputs, including a segmentation mask and an annotated image showing the detected cube.

### Outcome

Day 2 introduced a reusable detector that transforms a rendered image into a compact representation of the target: its pixel-space center and associated detection information.

## Day 3 — Pixel-to-World Position Estimation

### Objective

Convert the detected image coordinates into an estimated position in the MuJoCo world coordinate system.

### Implementation

The geometric projection is implemented in:

`vision/projection.py`

The function `pixel_to_world()` receives the MuJoCo model and data, camera name, detected pixel coordinates, image dimensions, and the known height of the target plane.

It uses the camera's vertical field of view to estimate the focal length in pixel units:

\[
f = \frac{H/2}{\tan(\mathrm{fovy}/2)}
\]

where:

- \(H\) is the image height in pixels.
- \(\mathrm{fovy}\) is the camera's vertical field of view in radians.
- \(f\) is the focal length expressed in pixels.

The principal point is approximated by the image center:

\[
c_x = \frac{W-1}{2},
\qquad
c_y = \frac{H-1}{2}
\]

where \(W\) and \(H\) are the image width and height.

The detected pixel coordinates \((u,v)\) are converted into a camera-frame ray:

\[
\mathbf{r}_{cam} =
\begin{bmatrix}
(u-c_x)/f\\
-(v-c_y)/f\\
-1
\end{bmatrix}
\]

The negative sign in the vertical image coordinate accounts for the fact that image coordinates increase downward, whereas the camera-frame vertical direction uses the opposite convention. The negative third component represents the viewing direction in the camera frame.

The ray is transformed into world coordinates using the camera rotation matrix obtained from MuJoCo:

\[
\mathbf{r}_{world}
=
R_{world,cam}\mathbf{r}_{cam}
\]

The camera's world position provides the ray origin, \(\mathbf{o}_{world}\).

The ray is then intersected with the known horizontal plane at height \(z=z_{plane}\). Its intersection parameter is:

\[
t =
\frac{z_{plane}-o_z}{r_z}
\]

The resulting world position is:

\[
\mathbf{p}_{world}
=
\mathbf{o}_{world}+t\mathbf{r}_{world}
\]

The implementation checks for invalid camera names, rays that are nearly parallel to the target plane, and intersections that lie behind the camera.

### Assumptions and limitations

The projection method assumes that:

- The camera calibration and field of view are known from the MuJoCo model.
- The detected pixel represents a point on the target plane.
- The target plane has a known world-coordinate height.
- Lens distortion is negligible or absent in the simulated camera model.

The current configuration uses a plane height of **0.04 m**, corresponding to the cube's center height in the configured scene. This allows the detected image center to be projected onto the horizontal plane passing through the cube center.

The method is suitable for this controlled simulation. For real-world deployment, camera calibration, lens distortion, uncertainty, occlusion, and deviations from the assumed target plane would need additional consideration.

### Quantitative validation

The validation script is:

`vision/position_validation.py`

It performs **20 randomized trials**. For each trial, it places the cube at a known planar position, renders the scene, detects the cube, estimates its world position, and compares the estimate with the simulated ground truth.

The reported Day 3 results were:

| Metric | Result |
|---|---:|
| Randomized trials | 20 |
| Successful detections | 20/20 |
| Missed detections | 0 |
| Mean planar position error | 3.4 mm |
| Maximum planar position error | 7.0 mm |
| Standard deviation of error | 1.8 mm |
| Negative detection test | Passed |

The planar error is calculated as the Euclidean distance between the estimated and true \(x,y\) coordinates:

\[
e_{xy}
=
\sqrt{(x_{est}-x_{true})^2+
(y_{est}-y_{true})^2}
\]

The error is reported in millimeters.

A negative detection test moves the cube outside the normal camera workspace and checks that the detector does not report a valid cube detection in the resulting image.

These results characterize the tested simulation setup and should not be interpreted as a guarantee of the same accuracy in a physical robot system.

## Day 4 — Configuration, Modular Scene Management, Testing, and Demo

### Objective

Improve maintainability, reduce duplicated configuration, automate regression testing, and prepare visual material documenting the pipeline.

### Centralized configuration

The file `vision/config.py` centralizes the main parameters used throughout the vision module:

- MuJoCo XML model path.
- Camera name.
- Target body name.
- Target plane height.
- Rendered image dimensions.
- HSV segmentation bounds.
- Minimum contour area.
- Morphological kernel size.

Centralizing these parameters reduces duplication and makes the pipeline easier to adapt to a different camera, target, or scene configuration.

### Scene-management utilities

The file `vision/scene.py` encapsulates the common operations required by the validation and demonstration scripts.

Its main functions are:

- `load_scene()`: loads the MuJoCo model, creates simulation data and a renderer, and resolves the target body's identifier.
- `place_cube()`: updates the target position and calls `mujoco.mj_forward()` to refresh derived simulation quantities.
- `render()`: updates the selected camera scene and returns the rendered image.

The renderer is explicitly closed after use to release graphics resources.

### Automated tests

The test suite is located in:

`tests/test_vision.py`

It verifies the key behaviors of the pipeline, including position estimation at several target positions and rejection of a negative detection case.

The tests use a maximum acceptable planar error of **0.015 m (15 mm)**. This tolerance is greater than the maximum error observed during the Day 3 validation and provides a practical margin for regression testing.

The test suite was executed successfully:

```text
4 passed in 1.01s
```

All four tests passed. This confirms that the tested detection and projection behaviors satisfy the assertions defined by the current test suite.

### Visual demonstration

The script `vision/demo.py` generates visual material illustrating the detection and position-estimation pipeline.

The intended outputs are:

- `docs/opencv_demo.gif`: an animated demonstration across multiple cube positions.
- `docs/opencv_overlay_example.png`: a representative annotated image.

The overlay illustrates the detected cube, its estimated world coordinates, and the position error. These outputs make it easier to inspect the detector's behavior and communicate the results without manually examining raw image arrays.

### Repository hygiene and dependencies

The project uses `requirements.txt` to record Python dependencies required by the vision pipeline and its tests, including OpenCV's headless package, ImageIO, and pytest, alongside the existing project dependencies.

Generated frames, Python bytecode, and pytest cache files should be excluded from version control through `.gitignore` where appropriate.

## Running the Pipeline

Run the commands from the repository root:

```bash
cd ~/mujoco_robotics
```

The vision package is invoked with Python's module syntax. This is important for scripts that import modules using package-qualified imports such as `from vision import config`.

### 1. Run the position validation

```bash
MUJOCO_GL=egl python3 -m vision.position_validation
```

This runs the randomized position-estimation experiment and prints the detection and error statistics.

### 2. Run the detection script

```bash
MUJOCO_GL=egl python3 -m vision.run_detection
```

This runs the image-detection workflow. The script's configured inputs and output paths determine which images are processed and saved.

### 3. Run the automated tests

```bash
MUJOCO_GL=egl python3 -m pytest -v tests/test_vision.py
```

The current test suite contains four tests.

### 4. Generate
