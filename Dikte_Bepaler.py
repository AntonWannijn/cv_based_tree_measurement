import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor

def triangulate_points(P1, P2, pts1, pts2):
    """Triangulate corresponding points between two views."""
    pts4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d_hom[:3] / pts4d_hom[3]).T
    return pts3d

def fit_plane_ransac(points_3d):
    """
    Fit a ground plane to a 3D point cloud using RANSAC.
    Plane: ax + by + cz + d = 0 --> normal = [a, b, c]
    """
    X = points_3d[:, :2]  # Use x and y
    y = points_3d[:, 2]   # Predict z

    ransac = RANSACRegressor()
    ransac.fit(X, y)

    a, b = ransac.estimator_.coef_
    c = -1
    d = ransac.estimator_.intercept_

    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    # Compute final plane distance (d_proj) from origin
    d_proj = -np.dot(normal, np.array([0, 0, d]))  # point on plane: (0, 0, d)

    return normal, d_proj

def estimate_ground_plane(K, frames, poses):
    """
    Estimate the ground plane using multiple frames and camera poses.

    Parameters:
        K (np.ndarray): Intrinsic matrix (3x3)
        frames (list of np.ndarray): List of grayscale frames
        poses (list of (R, t)): Camera poses (rotation + translation) per frame

    Returns:
        ground_normal (np.ndarray), ground_d (float)
    """
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Choose 2 distant frames for triangulation
    idx1, idx2 = 0, len(frames) - 1
    kp1, des1 = orb.detectAndCompute(frames[idx1], None)
    kp2, des2 = orb.detectAndCompute(frames[idx2], None)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Undistort and normalize to camera space
    pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2).T
    pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2).T

    R1, t1 = poses[idx1]
    R2, t2 = poses[idx2]

    # Projection matrices
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])

    # Triangulate 3D points
    pts_3d = triangulate_points(P1, P2, pts1.T, pts2.T)

    # Filter out points behind camera or too far
    z_valid = pts_3d[:, 2] > 0
    pts_3d = pts_3d[z_valid]

    # Fit ground plane
    ground_normal, ground_d = fit_plane_ransac(pts_3d)

    return ground_normal, ground_d


# --- Gebruikt voor het horizontaal veld voor te stellen ---

def project_plane_to_image(K, R, t, ground_normal, ground_d, height_above_ground=0.5, grid_size=1.0, grid_extent=5.0):
    """
    Projects a horizontal slicing plane 0.5m above the ground into image space.

    Parameters:
        K (np.ndarray): Intrinsic matrix (3x3)
        R (np.ndarray): Rotation matrix (3x3)
        t (np.ndarray): Translation vector (3x1)
        ground_normal (np.ndarray): Normal vector of the ground plane (3,)
        ground_d (float): Plane offset from origin (in meters)
        height_above_ground (float): Offset of slicing plane from ground
        grid_size (float): Spacing between grid points (meters)
        grid_extent (float): Half-width of grid (meters)

    Returns:
        List of projected 2D points (image coordinates)
    """

    # Offset the plane by height_above_ground in the normal direction
    new_d = ground_d - height_above_ground

    # Choose a coordinate system on the plane: center at origin
    # We'll make a square grid of 3D points on the plane for projection
    u = np.cross(ground_normal, [0, 0, 1])
    if np.linalg.norm(u) < 1e-3:
        u = np.array([1, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(ground_normal, u)
    v /= np.linalg.norm(v)

    # Create grid of points on the slicing plane in world coordinates
    points_3d = []
    for i in np.linspace(-grid_extent, grid_extent, int(2 * grid_extent / grid_size)):
        for j in np.linspace(-grid_extent, grid_extent, int(2 * grid_extent / grid_size)):
            point_on_plane = -(new_d * ground_normal) + i * u + j * v
            points_3d.append(point_on_plane)

    points_3d = np.array(points_3d)  # (N, 3)

    # Transform points to camera coordinates
    Rt = np.hstack([R, t.reshape(3, 1)])  # 3x4
    points_3d_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])  # Nx4
    points_cam = points_3d_hom @ Rt.T  # Nx3

    # Filter out points behind the camera
    mask = points_cam[:, 2] > 0
    points_cam = points_cam[mask]

    # Project to 2D image coordinates
    points_img = (K @ points_cam.T).T
    points_img = points_img[:, :2] / points_img[:, 2:3]  # Normalize

    return points_img.astype(int)

# --- Gebruikt voor de dikte van de boom te berekenen ---

def find_tree_width_at_height(contour, y_height_pixel):
    """
    Intersects a horizontal line with a tree contour at a given pixel height.

    Parameters:
        contour (np.ndarray): Nx2 array of (x, y) points defining tree contour.
        y_height_pixel (int): y-coordinate (in image) representing 0.5m height.

    Returns:
        tuple: (x1, x2), the two intersection x-coordinates, or None if <2 intersections.
    """
    intersections = []

    for i in range(len(contour)):
        pt1 = contour[i]
        pt2 = contour[(i + 1) % len(contour)]

        y1, y2 = pt1[1], pt2[1]

        # Check if the horizontal line crosses this edge
        if (y1 - y_height_pixel) * (y2 - y_height_pixel) < 0:
            # Linear interpolation to find x at the intersection point
            dy = y2 - y1
            dx = pt2[0] - pt1[0]
            if dy == 0:
                continue  # avoid division by zero
            t = (y_height_pixel - y1) / dy
            x_intersect = pt1[0] + t * dx
            intersections.append(x_intersect)

    if len(intersections) >= 2:
        intersections = sorted(intersections)
        return intersections[0], intersections[-1]
    else:
        return None  # Not enough intersections to define width