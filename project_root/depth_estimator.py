# depth_estimator.py
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, camera_matrix, dist_coeffs):
        self.K = camera_matrix
        self.dist = dist_coeffs
        self.orb = cv2.ORB_create(nfeatures=2000) # Increased features
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Store previous frame data for pose estimation
        self.prev_frame_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None

    def _find_and_match_features(self, frame1_gray, frame2_gray):
        kp1, des1 = self.orb.detectAndCompute(frame1_gray, None)
        kp2, des2 = self.orb.detectAndCompute(frame2_gray, None)

        if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
            return None, None, None, None

        matches = self.bf_matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Keep a reasonable number of good matches
        good_matches = matches[:max(50, int(len(matches) * 0.15))]


        if len(good_matches) < 5: # Minimum matches for findEssentialMat
            return None, None, None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        return pts1, pts2, kp1, kp2


    def estimate_camera_pose(self, current_frame_gray):
        """Estimates camera pose (R, t) relative to the previous frame."""
        if self.prev_frame_gray is None or self.K is None:
            # Initialize for the first frame
            self.prev_frame_gray = current_frame_gray
            self.prev_keypoints, self.prev_descriptors = self.orb.detectAndCompute(current_frame_gray, None)
            return None, None

        pts1_matched, pts2_matched, kp1, kp2 = self._find_and_match_features(self.prev_frame_gray, current_frame_gray)

        R, t = None, None
        if pts1_matched is not None and pts2_matched is not None and len(pts1_matched) >=5 :
            # Undistort points before finding Essential Matrix
            pts1_undistorted = cv2.undistortPoints(pts1_matched, self.K, self.dist, P=self.K)
            pts2_undistorted = cv2.undistortPoints(pts2_matched, self.K, self.dist, P=self.K)

            E, mask_e = cv2.findEssentialMat(pts1_undistorted, pts2_undistorted, self.K, 
                                             method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None and mask_e is not None:
                # Filter points using the mask from findEssentialMat
                # Ensure pts1_inliers and pts2_inliers are correctly shaped for recoverPose
                pts1_inliers = pts1_undistorted[mask_e.ravel() == 1]
                pts2_inliers = pts2_undistorted[mask_e.ravel() == 1]

                if len(pts1_inliers) >= 5: # Need at least 5 points for recoverPose
                    _, R_est, t_est, mask_rp = cv2.recoverPose(E, pts1_inliers, pts2_inliers, self.K)
                    
                    # Check if R_est and t_est are valid (e.g., not None, t_est has positive z component often)
                    # This simple check might need refinement based on motion direction
                    if R_est is not None and t_est is not None: # and np.mean(t_est) > 0: # Basic check
                        R = R_est
                        t = t_est
                else:
                    print("Not enough inliers after Essential Matrix for recoverPose.")
            else:
                print("Essential Matrix estimation failed.")
        else:
            print("Not enough matched features for pose estimation.")
            
        # Update previous frame data for the next iteration
        self.prev_frame_gray = current_frame_gray
        self.prev_keypoints = kp2 # Use keypoints from the current frame (which becomes previous)
        self.prev_descriptors = self.orb.detectAndCompute(current_frame_gray, None)[1] if kp2 else None


        return R, t

    def triangulate_points(self, R, t, tree_points_frame1, tree_points_frame2, baseline_scale=1.0):
        """
        Triangulates 3D points for trees.
        R, t: Rotation and translation from frame1 to frame2.
        tree_points_frame1/2: List of (u,v) tuples for a SINGLE tree in respective frames.
                              (Ensure these are corresponding points for the SAME tree)
        baseline_scale: The estimated actual distance the camera moved (magnitude of T).
        """
        if R is None or t is None or self.K is None:
            return None

        # Undistort tree points
        # Input needs to be like: np.array([[[u1,v1]], [[u2,v2]], ...], dtype=np.float32)
        if not isinstance(tree_points_frame1, np.ndarray) or tree_points_frame1.ndim != 3:
            tp1 = np.float32([[tree_points_frame1]]).reshape(-1,1,2)
        else:
            tp1 = tree_points_frame1
        if not isinstance(tree_points_frame2, np.ndarray) or tree_points_frame2.ndim != 3:
            tp2 = np.float32([[tree_points_frame2]]).reshape(-1,1,2)
        else:
            tp2 = tree_points_frame2
        
        tp1_undistorted = cv2.undistortPoints(tp1, self.K, self.dist, P=self.K)
        tp2_undistorted = cv2.undistortPoints(tp2, self.K, self.dist, P=self.K)

        # Projection matrices
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        
        # Scale the translation vector t
        T_scaled = t * baseline_scale 
        P2 = self.K @ np.hstack((R, T_scaled)) # T_scaled should be the actual translation vector

        # Triangulate (expects 2xN points)
        homogeneous_3D_point = cv2.triangulatePoints(P1, P2, tp1_undistorted.T, tp2_undistorted.T)
        
        if homogeneous_3D_point is not None and homogeneous_3D_point[3] != 0:
            euclidean_3D_point = homogeneous_3D_point[:3] / homogeneous_3D_point[3]
            # Check if the point is in front of both cameras
            # Point in frame1 coords: P_f1 = euclidean_3D_point
            # Point in frame2 coords: P_f2 = R @ P_f1 + T_scaled
            # if euclidean_3D_point[2] > 0 and (R @ euclidean_3D_point + T_scaled)[2] > 0:
            # A simpler check is often if the z-depth is positive.
            if euclidean_3D_point[2][0] > 0: # Z-coordinate (depth)
                 return euclidean_3D_point.flatten() # Returns (X, Y, Z)
        return None

    def estimate_baseline_scale(self, R, t, frame_rate, vehicle_speed_mps=None):
        """
        Estimates the baseline scale.
        THIS IS A PLACEHOLDER. Real scale estimation is complex.
        Args:
            vehicle_speed_mps (float, optional): Speed of the camera in meters/second.
            frame_rate (float): Processing frame rate (frames per second).
        Returns:
            float: Estimated baseline scale. Default is 1.0 (unit vector).
        """
        if vehicle_speed_mps is not None and frame_rate > 0:
            # Assuming pose estimation is done between consecutive processed frames
            time_diff_seconds = 1.0 / frame_rate 
            return vehicle_speed_mps * time_diff_seconds
        return 1.0 # Default scale (results in scaled 3D reconstruction)


if __name__ == '__main__':
    # This is a complex module to test standalone without a video sequence
    # and camera calibration.
    # You would typically call estimate_camera_pose in a loop for a video,
    # then use the R, t with tracked tree points for triangulate_points.
    
    # Dummy K for syntax check
    K_dummy = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    dist_dummy = np.zeros(5, dtype=np.float32)
    depth_est = DepthEstimator(K_dummy, dist_dummy)

    # Dummy frames
    frame1_gray_dummy = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    frame2_gray_dummy = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

    # First call to initialize prev_frame_gray etc.
    depth_est.estimate_camera_pose(frame1_gray_dummy) 
    R_test, t_test = depth_est.estimate_camera_pose(frame2_gray_dummy)

    if R_test is not None and t_test is not None:
        print("Estimated Pose R:\n", R_test, "\nt:\n", t_test)
        # Dummy tree points (center of a bounding box in each frame)
        tree_pt_f1 = (300, 200) 
        tree_pt_f2 = (310, 205) # Slightly shifted
        
        # Assume a baseline scale (e.g., camera moved 0.1 meters)
        baseline = depth_est.estimate_baseline_scale(R_test, t_test, frame_rate=10, vehicle_speed_mps=1.0) # 1m/s, 10fps -> 0.1m baseline
        print(f"Estimated baseline: {baseline} m")

        point_3d = depth_est.triangulate_points(R_test, t_test, tree_pt_f1, tree_pt_f2, baseline_scale=baseline)
        if point_3d is not None:
            print("Triangulated 3D point:", point_3d, " (X, Y, Z in meters from camera 1)")
            print(f"Depth (Z): {point_3d[2]:.2f} m")
        else:
            print("Triangulation failed for dummy points.")
    else:
        print("Pose estimation failed for dummy frames.")