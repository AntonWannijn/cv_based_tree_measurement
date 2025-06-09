import cv2
import numpy as np





def extract_and_match_features(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe’s ratio test
    good = []
    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
            good.append(m)

    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)

    match_vis = cv2.drawMatches(
        img1, kp1, img2, kp2, good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return pts1, pts2, match_vis


def compute_essential_matrix(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]
    return E, inliers1, inliers2

def extract_and_match_features(img1, img2):
    """
    Detecteert ORB-features en matcht ze tussen twee beelden.
    
    Returns:
        pts1 (np.ndarray): Matchende keypoints in beeld 1
        pts2 (np.ndarray): Matchende keypoints in beeld 2
        matches_img (np.ndarray): Visualisatiebeeld van matches
    """
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    pts1, pts2, good_matches = [], [], []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
            good_matches.append(m)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    matches_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, flags=2)

    return pts1, pts2, matches_img


def compute_essential_matrix(pts1, pts2, K):
    """
    Berekent de fundamentele en essentiële matrix.

    Returns:
        E (np.ndarray): Essentiële matrix
        F (np.ndarray): Fundamentele matrix
        mask (np.ndarray): Masker van inliers
    """
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    E = K.T @ F @ K
    return E, F, mask


def recover_camera_pose(E, pts1, pts2, K):
    """
    Haalt rotatie en translatie uit de essentiële matrix.

    Returns:
        R (np.ndarray): 3x3 rotatiematrix
        t (np.ndarray): 3x1 translatievector
        mask (np.ndarray): mask van inliers gebruikt door recoverPose
    """
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask

