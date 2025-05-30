# camera_calibration.py
import numpy as np
import os

def load_calibration_data(calibration_file_path="calibration_data.npz"):
    """
    Loads camera calibration data (matrix and distortion coefficients).

    Args:
        calibration_file_path (str): Path to the .npz file with calibration data.

    Returns:
        tuple: (camera_matrix, dist_coeffs) or (None, None) if file not found.
    """
    if not os.path.exists(calibration_file_path):
        print(f"Error: Calibration file not found at {calibration_file_path}")
        print("Please run kalibreren.py first to generate this file.")
        return None, None
        
    try:
        data = np.load(calibration_file_path)
        camera_matrix = data['mtx']
        dist_coeffs = data['dist']
        print("Camera calibration data loaded successfully.")
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return None, None

if __name__ == '__main__':
    # Example usage:
    # Make sure 'calibration_data_10x6.npz' (or your chosen name) is in the same directory
    # or provide the full path. This path should match the output of your kalibreren.py.
    # For example, if kalibreren.py saves to 'D:\\calibration_data_10x6.npz', use that path.
    # For testing, assuming it's in the current directory:
    test_calib_file = 'calibration_data.npz' # Or 'calibration_data.npz'
    if not os.path.exists(test_calib_file) and os.path.exists('calibration_data.npz'):
        test_calib_file = 'calibration_data.npz'
    
    mtx, dist = load_calibration_data(test_calib_file)
    if mtx is not None:
        print("Camera Matrix (K):\n", mtx)
        print("Distortion Coefficients:\n", dist)