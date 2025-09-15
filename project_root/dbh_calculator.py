# dbh_calculator.py
import cv2
import numpy as np

# --- Adapting core logic from your calc_width.py (remains the same) ---
def preprocess_roi_for_edges(roi_bgr, blur_kernel_size=9):
    if roi_bgr is None or roi_bgr.size == 0: return None
    if len(roi_bgr.shape) == 3 and roi_bgr.shape[2] == 3:
        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    elif len(roi_bgr.shape) == 2: gray_roi = roi_bgr
    else: return None
    mask = cv2.medianBlur(gray_roi, blur_kernel_size)
    mask = cv2.Canny(mask, 50, 150)
    return mask

def count_left_edge(row_of_mask):
    counter = 0
    for i in range(len(row_of_mask)):
        if row_of_mask[i] == 255: break
        counter += 1
    return counter

def count_right_edge(row_of_mask):
    w = len(row_of_mask) - 1
    counter = w
    for i in range(w, -1, -1):
        if row_of_mask[i] == 255: break
        counter -= 1
    return counter

def find_trunk_pixel_width_at_row(roi_bgr_for_tree, target_row_in_roi):
    if roi_bgr_for_tree is None or roi_bgr_for_tree.size == 0: return 0
    edge_mask_of_roi = preprocess_roi_for_edges(roi_bgr_for_tree)
    if edge_mask_of_roi is None: return 0
    roi_h, roi_w = edge_mask_of_roi.shape[:2]
    if not (0 <= target_row_in_roi < roi_h): return 0
    selected_row_from_mask = edge_mask_of_roi[target_row_in_roi, :]
    left_px = count_left_edge(selected_row_from_mask)
    right_px = count_right_edge(selected_row_from_mask)
    if right_px <= left_px: return 0
    pixel_width = right_px - left_px
    return pixel_width if pixel_width > 0 else 0
# --- End of adapted calc_width.py logic ---


class DBHCalculator:
    def __init__(self, camera_matrix):
        """
        Args:
            camera_matrix (np.ndarray): The intrinsic camera matrix K.
        """
        self.K = camera_matrix
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        # self.cx = camera_matrix[0, 2] # Not directly used in this version of DBH calc
        # self.cy = camera_matrix[1, 2] # Not directly used here
        self.DBH_standard_height_m = 1.3 # Standard height for DBH measurement from ground

    def find_tree_base_pixel_in_image(self, tree_bbox, ground_mask_full_image):
        """
        Finds the pixel row (v_base) where the tree base meets the ground.
        Args:
            tree_bbox (tuple): (x1, y1, x2, y2) of the tree in the full image.
            ground_mask_full_image (np.ndarray): Binary mask of the ground in the full image.
        Returns:
            int: The v-coordinate (pixel row from top, $v_{base}$) of the tree base,
                 or None if not found.
        """
        x1, y1, x2, y2 = tree_bbox

        # Consider a central vertical strip of the bounding box to find the base
        # This helps avoid edges of the bbox that might not be on the trunk
        search_x_start = x1 + (x2 - x1) // 4
        search_x_end = x2 - (x2 - x1) // 4
        
        if search_x_start >= search_x_end : search_x_start = x1; search_x_end = x2 # Failsafe for very thin bboxes

        # Search from bottom of bbox  within the ground mask
        # We expect the tree base to be at the highest y-value (bottom of image)
        # of the trunk that is still on the ground.
        # Or, the lowest y-value of the trunk that touches the non-ground part, if viewed from above.
        # Let's find the lowest point of the tree bbox that intersects the ground.
        
        # Iterate from the bottom of the bounding box (y2) upwards.
        # The first row (from bottom) within the bbox that is on the ground is a candidate.
        # To be more robust, we can check a small region or the lowest average point.
        
        # For simplicity: find the lowest y-coordinate (max y) within the bbox's
        # central x-strip that is classified as ground.
        v_base_candidate = None
        for v_row in range(y2 -1 , y1 -1, -1): # Iterate from bottom of bbox upwards
            # Check if any pixel in the horizontal slice of the central strip at v_row is ground
            is_on_ground = False
            if 0 <= v_row < ground_mask_full_image.shape[0]: # check bounds
                for u_col in range(search_x_start, search_x_end):
                    if 0 <= u_col < ground_mask_full_image.shape[1]: # check bounds
                        if ground_mask_full_image[v_row, u_col] == 255: # 255 means ground
                            is_on_ground = True
                            break
            if is_on_ground:
                v_base_candidate = v_row
                break # Found the lowest ground point along the trunk base

        return v_base_candidate


    def get_dbh_pixel_row_from_base(self, v_base_image, tree_distance_z_m):
        """
        Calculates the pixel row for DBH measurement, given the tree's base row in image
        and its distance.
        Args:
            v_base_image (int): Pixel row of the tree's base in the full image.
            tree_distance_z_m (float): Distance to the tree (Z-coordinate) in meters.
                                     (This should be distance to the base or approximately to the trunk)
        Returns:
            int: The v-coordinate (pixel row) for DBH measurement, or None.
        """
        if v_base_image is None or tree_distance_z_m <= 0:
            return None

        # delta_v = (fy * WorldHeight) / Z_distance
        delta_v_for_dbh = (self.fy * self.DBH_standard_height_m) / tree_distance_z_m
        
        # v_DBH is above v_base (smaller v_coordinate if y points down)
        v_dbh_pixel = int(v_base_image - delta_v_for_dbh)
        return v_dbh_pixel

    def calculate_dbh_cm(self, tree_roi_bgr, tree_bbox_full_image, ground_mask_full_image, tree_distance_z_m):
        """
        Calculates the DBH in centimeters using ground segmentation to find tree base.
        Args:
            tree_roi_bgr (np.ndarray): The BGR ROI of the detected tree.
            tree_bbox_full_image (tuple): (x1,y1,x2,y2) of tree in full image.
            ground_mask_full_image (np.ndarray): Binary mask of ground in full image.
            tree_distance_z_m (float): Distance to the tree (Z) in meters.
                                     (Approximation: distance to the center or base of the tree)
        Returns:
            float: Estimated DBH in cm, or 0 if calculation fails.
        """
        if tree_distance_z_m <= 0 or tree_roi_bgr is None or tree_roi_bgr.size == 0:
            return 0

        x1_full, y1_full, _, _ = tree_bbox_full_image

        # 1. Find tree base pixel row in the full image
        v_base_full_image = self.find_tree_base_pixel_in_image(tree_bbox_full_image, ground_mask_full_image)
        if v_base_full_image is None:
            # print("Could not determine tree base on ground.")
            return 0

        # 2. Calculate the DBH pixel row in the full image
        v_dbh_full_image = self.get_dbh_pixel_row_from_base(v_base_full_image, tree_distance_z_m)
        if v_dbh_full_image is None:
            return 0

        # 3. Convert the full image DBH row to a row within the ROI
        # The ROI's top edge corresponds to y1_full in the original image.
        target_row_in_roi = int(v_dbh_full_image - y1_full)
        
        if not (0 <= target_row_in_roi < tree_roi_bgr.shape[0]):
            # print(f"DBH measurement row {target_row_in_roi} (orig: {v_dbh_full_image}) is outside ROI {tree_roi_bgr.shape[0]} height.")
            return 0

        # 4. Measure pixel width at that row in the ROI
        pixel_width_at_dbh = find_trunk_pixel_width_at_row(tree_roi_bgr, target_row_in_roi)

        if pixel_width_at_dbh <= 0:
            # print(f"Could not measure valid pixel width at DBH row for tree at Z={tree_distance_z_m:.2f}m.")
            return 0
            
        # 5. Convert pixel width to real-world width
        real_width_m = (pixel_width_at_dbh * tree_distance_z_m) / self.fx
        dbh_cm = real_width_m * 100
        
        return dbh_cm


if __name__ == '__main__':
    K_dummy = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    dbh_calc = DBHCalculator(K_dummy)

    # Simulate image, bbox, ground mask
    img_h, img_w = 480, 640
    dummy_ground_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    dummy_ground_mask[img_h - 100:, :] = 255 # Last 100 rows are ground

    tree_bbox = (300, 200, 330, img_h - 50) # x1, y1, x2, y2 (tree base is on ground)
    x1,y1,x2,y2 = tree_bbox
    
    # Simulate ROI
    dummy_roi = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
    cv2.rectangle(dummy_roi, (5,0), (dummy_roi.shape[1]-5, dummy_roi.shape[0]-1), (70,100,120), -1)


    v_base = dbh_calc.find_tree_base_pixel_in_image(tree_bbox, dummy_ground_mask)
    print(f"Found v_base at: {v_base}") # Should be around y2-1 = (img_h - 50) - 1

    if v_base:
        Z_tree_m = 5.0
        v_dbh = dbh_calc.get_dbh_pixel_row_from_base(v_base, Z_tree_m)
        print(f"For tree at {Z_tree_m}m, with base at {v_base}, DBH measurement row: {v_dbh}")

        estimated_dbh = dbh_calc.calculate_dbh_cm(dummy_roi, tree_bbox, dummy_ground_mask, Z_tree_m)
        print(f"Estimated DBH for tree at {Z_tree_m}m: {estimated_dbh:.2f} cm")

        # Test case where base is not found
        no_ground_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        estimated_dbh_no_ground = dbh_calc.calculate_dbh_cm(dummy_roi, tree_bbox, no_ground_mask, Z_tree_m)
        print(f"Estimated DBH (no ground): {estimated_dbh_no_ground:.2f} cm") # Should be 0