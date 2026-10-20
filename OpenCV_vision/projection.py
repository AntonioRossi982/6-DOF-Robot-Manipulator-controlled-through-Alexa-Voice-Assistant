
import numpy as np
import mujoco


def pixel_to_world(model, data, cam_name, u, v, width, height, z_plane):
    """
    Converts a pixel (u, v) to world coordinates,
    by intersecting the camera ray with the plane z = z_plane.

    u, v: pixel coordinates
    width, height: image dimensions in pixels
    z_plane: known elevation of the point to be estimated, in meters
    """
    cam_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name
    )

    if cam_id < 0:
        raise ValueError(f"Camera not found: {cam_name}")

    # Intrinsic parameters of the camera
    fovy = np.deg2rad(model.cam_fovy[cam_id]) # conversion degrees - radians
    f = (height / 2) / np.tan(fovy / 2) # evaluation of focal length in pixels.

    # Center of the image (coordinates)
    cx = (width - 1) / 2
    cy = (height - 1) / 2

    # Ray expressed in the camera coordinate system
    ray_cam = np.array([
        (u - cx) / f,
        -(v - cy) / f,
        -1.0
    ])

    # Transformation from the camera coordinate system to the world coordinate system
    rotation = data.cam_xmat[cam_id].reshape(3, 3)
    ray_world = rotation @ ray_cam # transformation into the MuJoCo global world system
    origin = data.cam_xpos[cam_id] # Camera position in the world coordinate system

    # Avoid division by a ray almost parallel to the plane
    if abs(ray_world[2]) < 1e-9:
        raise ValueError("The ray is parallel to the requested plane")

    # Intersection with the plane z = z_plane (solving for t in the ray equation)
    t = (z_plane - origin[2]) / ray_world[2]

    if t < 0:
        raise ValueError("The plane intersects the ray behind the camera")

    # Compute the intersection point in the world coordinate system
    point_world = origin + t * ray_world
    
    return point_world