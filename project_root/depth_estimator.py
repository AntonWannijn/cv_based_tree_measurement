# depth_estimator.py
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, camera_matrix, dist_coeffs):
        self.K = camera_matrix
        self.dist = dist_coeffs
        
        # --- Change 1: Initialize SIFT detector instead of ORB ---
        self.sift = cv2.SIFT_create()

        # --- Change 2: Set up FLANN matcher for SIFT descriptors ---
        # SIFT descriptors are float32, so we use FLANN's KDTree index
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50) # or more checks for higher accuracy
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Store previous frame data for pose estimation
        self.prev_frame_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None

    def _find_and_match_features(self, frame1_gray, frame2_gray):
        # Use SIFT to detect keypoints and compute descriptors
        kp1, des1 = self.sift.detectAndCompute(frame1_gray, None)
        kp2, des2 = self.sift.detectAndCompute(frame2_gray, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return None, None, None, None

        # Use FLANN matcher to find the 2 best matches for each descriptor
        matches = self.flann_matcher.knnMatch(des1, des2, k=2)
        
        # --- Change 3: Apply Lowe's ratio test to filter for good matches ---
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance: # Ratio test
                    good_matches.append(m)
        except ValueError:
            # This can happen if knnMatch doesn't return enough matches for some descriptors
            pass


        # Need at least 5 points for findEssentialMat
        if len(good_matches) < 5: 
            return None, None, None, None

        # Extract location of good matches
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        return pts1, pts2, kp1, kp2


    def estimate_camera_pose(self, current_frame_gray):
        """Estimates camera pose (R, t) relative to the previous frame."""
        if self.prev_frame_gray is None or self.K is None:
            # Initialize for the first frame
            self.prev_frame_gray = current_frame_gray
            self.prev_keypoints, self.prev_descriptors = self.sift.detectAndCompute(current_frame_gray, None)
            return None, None

        pts1_matched, pts2_matched, kp1, kp2 = self._find_and_match_features(self.prev_frame_gray, current_frame_gray)

        R, t = None, None
        if pts1_matched is not None and pts2_matched is not None and len(pts1_matched) >=5 :
            pts1_undistorted = cv2.undistortPoints(pts1_matched, self.K, self.dist, P=self.K)
            pts2_undistorted = cv2.undistortPoints(pts2_matched, self.K, self.dist, P=self.K)

            E, mask_e = cv2.findEssentialMat(pts1_undistorted, pts2_undistorted, self.K, 
                                             method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None and mask_e is not None:
                pts1_inliers = pts1_undistorted[mask_e.ravel() == 1]
                pts2_inliers = pts2_undistorted[mask_e.ravel() == 1]

                if len(pts1_inliers) >= 5:
                    _, R_est, t_est, mask_rp = cv2.recoverPose(E, pts1_inliers, pts2_inliers, self.K)
                    if R_est is not None and t_est is not None:
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
        # For SIFT, kp2 might not be defined if matching fails early, so check for it.
        self.prev_keypoints = kp2 if 'kp2' in locals() and kp2 is not None else None
        if self.prev_keypoints is not None:
             self.prev_descriptors = self.sift.compute(current_frame_gray, self.prev_keypoints)[1]
        else:
             self.prev_descriptors = None

        return R, t

    def triangulate_points(self, R, t_unit, tree_points_frame1, tree_points_frame2, baseline_scale=1.0):
        # ... (This function does not need to be changed as it's independent of the feature type) ...
        if R is None or t_unit is None or self.K is None:
            return None
        tp1 = np.float32([[tree_points_frame1]]).reshape(-1,1,2)
        tp2 = np.float32([[tree_points_frame2]]).reshape(-1,1,2)
        tp1_undistorted = cv2.undistortPoints(tp1, self.K, self.dist, P=self.K)
        tp2_undistorted = cv2.undistortPoints(tp2, self.K, self.dist, P=self.K)
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        T_metric = t_unit.reshape(3,1) * baseline_scale 
        P2 = self.K @ np.hstack((R, T_metric)) 
        homogeneous_3D_point = cv2.triangulatePoints(P1, P2, tp1_undistorted.reshape(2,1), tp2_undistorted.reshape(2,1))
        if homogeneous_3D_point is not None and homogeneous_3D_point[3][0] != 0:
            euclidean_3D_point = homogeneous_3D_point[:3] / homogeneous_3D_point[3]
            if euclidean_3D_point[2][0] > 0:
                point_in_cam2_coords = R @ euclidean_3D_point + T_metric
                if point_in_cam2_coords[2][0] > 0:
                    return euclidean_3D_point.flatten()
        return None

    def estimate_baseline_scale(self, R, t, frame_rate, vehicle_speed_mps=None):
        # ... (This function does not need to be changed) ...
        if vehicle_speed_mps is not None and frame_rate > 0:
            time_diff_seconds = 1.0 / frame_rate 
            return vehicle_speed_mps * time_diff_seconds
        return 1.0