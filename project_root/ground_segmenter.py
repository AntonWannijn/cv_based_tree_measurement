# ground_segmenter.py
import cv2
import numpy as np

class GroundSegmenter:
    def __init__(self):
        """
        Initializes the ground segmenter.
        For this basic version, we'll define a sample color range for the ground.
        These HSV ranges would need to be tuned for your specific road/ground color.
        Example: Detecting grayish road.
        """
        # Placeholder: Simple HSV color range for "road-like" color.
        # This is highly dependent on your specific video conditions.
        # You'll likely need a much more robust method.
        self.lower_hsv_ground = np.array([0, 0, 50])    # Lower bound for HSV (e.g., dark grays)
        self.upper_hsv_ground = np.array([180, 50, 200]) # Upper bound for HSV (e.g., light grays)
        print("Basic GroundSegmenter initialized (Color-based placeholder).")

    def segment_ground(self, frame):
        """
        Segments the ground in the given frame.

        Args:
            frame (numpy.ndarray): The input BGR image/frame.

        Returns:
            numpy.ndarray: A binary mask where white pixels (255) represent the ground,
                           and black pixels (0) represent non-ground.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask for the ground color
        ground_mask = cv2.inRange(hsv_frame, self.lower_hsv_ground, self.upper_hsv_ground)
        
        # Optional: Apply some morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_OPEN, kernel)
        ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, kernel)
        
        return ground_mask

if __name__ == '__main__':
    # Example Usage (requires an image)
    segmenter = GroundSegmenter()
    
    # Create a dummy frame with a gray bottom half for testing
    frame_h, frame_w = 480, 640
    dummy_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    dummy_frame[frame_h//2:, :, :] = [100, 100, 100] # Gray bottom half (road)
    dummy_frame[:frame_h//2, :, :] = [50, 150, 50]   # Greenish top half (not road)


    if dummy_frame is not None:
        mask = segmenter.segment_ground(dummy_frame)
        # cv2.imshow("Original Frame", dummy_frame)
        # cv2.imshow("Ground Mask", mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print("Ground segmentation example run. Check a real image for HSV tuning.")
    else:
        print("Failed to create/load test image.")