import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import time # For simple FPS calculation/timing if needed

from camera_calibration import load_calibration_data
from tree_detector import TreeDetector
from tree_tracker import BasicTracker # Replace with a better tracker if possible
from depth_estimator import DepthEstimator
from ground_segmenter import GroundSegmenter
from dbh_calculator import DBHCalculator
from visualizer import Visualizer
from utils import clear_or_init_csv, log_results_to_csv

def main():
    # --- Configuration ---
    video_source = "input_video.mp4" # Or 0 for webcam, or your actual video file path
    # Example: video_source = r"path/to/your/video.mp4"
    # When running the file using CMD prompt: first use this command:
    # set OPENCV_FFMPEG_READ_ATTEMPTS=50000
    # This is to increase the amount of FFMPEG read attempts, the standard value is 4096
    # Since our video is so large we need to increase the attempts
    
    yolo_model_path = "best.pt"
    # Ensure this path matches the output of your kalibreren.py
    calibration_file = 'calibration_data.npz' 
    
    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)
    csv_filepath = os.path.join(output_dir, "tree_measurements_final.csv")
    # Added v_base, v_dbh for debugging, world_y_m
    csv_header = ['frame_id', 'tree_id', 
                  'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 
                  'world_x_m', 'world_y_m', 'world_z_m', 
                  'v_base_px', 'v_dbh_px', 'dbh_cm']
    
    output_video_path = os.path.join(output_dir, "annotated_video_1fps_final.mp4")

    # Critical assumptions for metric scale
    vehicle_speed_kmh = 30.0  # ASSUMED average speed in km/h. Adjust this!
    vehicle_speed_mps = vehicle_speed_kmh * 1000 / 3600

    # --- Initialization ---
    camera_matrix, dist_coeffs = load_calibration_data(calibration_file)
    if camera_matrix is None:
        print("Exiting due to camera calibration data error.")
        return

    detector = TreeDetector(model_path=yolo_model_path)
    if detector.model is None:
        print("Exiting due to YOLO model loading error.")
        return
        
    tracker = BasicTracker(iou_threshold=0.2, max_staleness=10) # Tune IoU and staleness
    depth_module = DepthEstimator(camera_matrix, dist_coeffs)
    ground_module = GroundSegmenter() 
    dbh_module = DBHCalculator(camera_matrix)
    visualizer = Visualizer(map_size=(1000,1000), map_scale_pixels_per_meter=10) # Adjusted map params

    # Ensure the video file exists if it's not a webcam
    if not isinstance(video_source, int) and not os.path.exists(video_source):
        print(f"Error: Video file not found at {video_source}")
        if video_source == "input_video.mp4": # Create dummy if default name and not found
            print("Creating a dummy 'input_video.mp4' for testing. Replace with your actual video.")
            fourcc_dummy = cv2.VideoWriter_fourcc(*'mp4v')
            dummy_out = cv2.VideoWriter(video_source, fourcc_dummy, 1.0, (640, 480))
            for _ in range(30): dummy_out.write(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            dummy_out.release()
        else:
            return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps == 0 or source_fps is None: 
        print("Warning: Could not read source FPS. Assuming 30 FPS.")
        source_fps = 30.0
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # For 1FPS output video, as requested
    out_video_1fps = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1.0, (frame_width, frame_height))
    clear_or_init_csv(csv_filepath, csv_header) # Prepare CSV file

    # --- Main Loop State Variables ---
    frame_id_counter = 0
    # Global camera pose in world (starts at origin, identity rotation)
    # T_world_cam: transforms points from camera coordinates to world coordinates
    world_T_camera = np.eye(4) 
    all_tree_world_positions = {} 
    
    # Store 2D points and corresponding world pose of camera for tracked trees from the *previous* valid SfM frame
    prev_sfm_frame_data = {
        "tree_points_2d": {}, # {track_id: (u,v)} in prev_sfm_frame
        "world_T_camera": np.eye(4) # World pose of the camera at prev_sfm_frame
    }
    
    output_frame_write_counter = 0
    frames_to_process_for_1fps_output = int(source_fps) if source_fps >=1 else 1


    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        
        frame_id_counter += 1
        start_time_frame = time.time() # For timing
        
        frame_to_process = frame.copy()
        gray_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)

        # 0. Ground Segmentation
        ground_mask = ground_module.segment_ground(frame_to_process)

        # 1. Tree Detection
        detections = detector.detect_trees(frame_to_process, conf_threshold=0.3) # Increased confidence slightly

        # 2. Tree Tracking
        tracked_objects = tracker.update(detections) # List of {'bbox':..., 'track_id':...}

        # 3. Depth Estimation & 3D Localization (SfM)
        # R_curr_prev, t_curr_prev_unit: transform points from prev_sfm_frame_coords to current_frame_coords
        R_curr_prev, t_curr_prev_unit = depth_module.estimate_camera_pose(gray_frame) 
        
        current_dbh_results = {} # {track_id: dbh_cm} for this frame's annotations
        current_frame_debug_info = {} # {track_id: {'v_base':vb, 'v_dbh':vd}}
        current_frame_tree_data_for_csv = []

        # Store current 2D points for next iteration's triangulation
        current_frame_points_2d_for_tracking = {}
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            # Use bottom-center of bbox as the representative point for the tree
            u_curr, v_curr = (x1 + x2) // 2, y2 
            current_frame_points_2d_for_tracking[obj['track_id']] = (u_curr, v_curr)


        if R_curr_prev is not None and t_curr_prev_unit is not None:
            # Calculate metric baseline for this frame pair
            time_between_sfm_frames_s = 1.0 / source_fps # Assuming SfM uses consecutive frames
            baseline_m = vehicle_speed_mps * time_between_sfm_frames_s
            
            if baseline_m < 0.01: # Avoid extremely small baselines
                print(f"Frame {frame_id_counter}: Warning - Baseline ({baseline_m:.4f}m) too small. Skipping SfM update this frame.")
            else:
                # Transformation from previous SfM camera to current camera
                T_curr_prev = np.eye(4)
                T_curr_prev[:3, :3] = R_curr_prev
                t_metric = t_curr_prev_unit.reshape(3, 1) * baseline_m
                T_curr_prev[:3, 3] = t_metric.flatten()

                # Update global camera pose: world_T_current = world_T_previous @ inv(T_current_previous)
                try:
                    world_T_camera = prev_sfm_frame_data["world_T_camera"] @ np.linalg.inv(T_curr_prev)
                except np.linalg.LinAlgError:
                    print(f"Frame {frame_id_counter}: Singular matrix in pose update. Resetting pose or skipping.")
                    # Potentially reset world_T_camera or skip this SfM update
                    # For simplicity, we continue but this indicates an issue.
                    R_curr_prev = None # Invalidate this SfM step

        # Perform triangulation if we had a valid pose estimation in this step (R_curr_prev is not None)
        if R_curr_prev is not None and baseline_m >=0.01: # Check baseline again
            for obj in tracked_objects:
                track_id = obj['track_id']
                bbox_full = obj['bbox']
                x1, y1, x2, y2 = bbox_full
                
                # (u_curr, v_curr) from current_frame_points_2d_for_tracking
                if track_id not in current_frame_points_2d_for_tracking: continue
                u_curr, v_curr = current_frame_points_2d_for_tracking[track_id]

                v_base_px, v_dbh_px, dbh_val = None, None, 0.0

                if track_id in prev_sfm_frame_data["tree_points_2d"]:
                    u_prev, v_prev = prev_sfm_frame_data["tree_points_2d"][track_id]
                    
                    # Triangulate point. Result is in coords of the *first* camera of the pair (prev_sfm_frame_data's camera)
                    point_3d_in_prev_cam_coords = depth_module.triangulate_points(
                        R_curr_prev, t_curr_prev_unit, # R, t_unit of current relative to previous
                        (u_prev, v_prev), (u_curr, v_curr), 
                        baseline_scale=baseline_m
                    )
                    
                    if point_3d_in_prev_cam_coords is not None:
                        Z_tree_m_from_prev_cam = point_3d_in_prev_cam_coords[2]

                        # Transform point from prev_sfm_camera's frame to world frame
                        point_3d_prev_cam_homogeneous = np.append(point_3d_in_prev_cam_coords, 1).reshape(4, 1)
                        tree_world_coord_homogeneous = prev_sfm_frame_data["world_T_camera"] @ point_3d_prev_cam_homogeneous
                        tree_world_coord = tree_world_coord_homogeneous[:3].flatten()
                        
                        # Store/update world position using current track_id
                        # Only update if Z is reasonable (e.g. positive and not too far)
                        if Z_tree_m_from_prev_cam > 0.1 and Z_tree_m_from_prev_cam < 100: # Basic filter
                             all_tree_world_positions[track_id] = tree_world_coord
                        
                        # 4. DBH Calculation
                        # Use current frame's ROI and bbox, but Z_tree_m from prev_cam's perspective.
                        # This assumes tree hasn't moved much relative to camera between the two frames for DBH appearance.
                        tree_roi_bgr = frame_to_process[y1:y2, x1:x2]
                        
                        v_base_px = dbh_module.find_tree_base_pixel_in_image(bbox_full, ground_mask)
                        if v_base_px:
                            # Use Z_tree_m_from_prev_cam as the distance for DBH height projection
                            v_dbh_px = dbh_module.get_dbh_pixel_row_from_base(v_base_px, Z_tree_m_from_prev_cam)
                        
                        dbh_val = dbh_module.calculate_dbh_cm(tree_roi_bgr, bbox_full, ground_mask, Z_tree_m_from_prev_cam)
                        
                        if dbh_val > 0:
                            current_dbh_results[track_id] = dbh_val
                        
                        current_frame_debug_info[track_id] = {'v_base': v_base_px, 'v_dbh': v_dbh_px}
                        current_frame_tree_data_for_csv.append({
                            'frame_id': frame_id_counter, 'tree_id': track_id,
                            'bbox_x1': x1, 'bbox_y1': y1, 'bbox_x2': x2, 'bbox_y2': y2,
                            'world_x_m': tree_world_coord[0], 'world_y_m': tree_world_coord[1], 'world_z_m': tree_world_coord[2],
                            'v_base_px': v_base_px, 'v_dbh_px': v_dbh_px,
                            'dbh_cm': dbh_val if dbh_val > 0 else None
                        })
            
            # After processing all trees for this SfM pair, update prev_sfm_frame_data for the *next* iteration
            prev_sfm_frame_data["tree_points_2d"] = current_frame_points_2d_for_tracking.copy()
            prev_sfm_frame_data["world_T_camera"] = world_T_camera.copy() # Store the pose of the current frame
        
        # If SfM failed, prev_sfm_frame_data is not updated with current points, so next iteration won't use them.
        # Consider if prev_sfm_frame_data should be cleared or how to handle multiple failed SfM steps.
        # For now, if SfM fails, `prev_sfm_frame_data` just retains its older values. `depth_module.prev_frame_gray` handles its own state.


        # 5. Visualization
        annotated_frame = visualizer.draw_on_frame(frame_to_process.copy(), tracked_objects, current_dbh_results)
        for obj in tracked_objects: # Add debug lines for v_base, v_dbh
            tid = obj['track_id']
            if tid in current_frame_debug_info:
                vb = current_frame_debug_info[tid].get('v_base')
                vd = current_frame_debug_info[tid].get('v_dbh')
                if vb is not None and 0 <= vb < frame_height :
                    cv2.line(annotated_frame, (obj['bbox'][0], vb), (obj['bbox'][2], vb), (0, 255, 255), 1) # Yellow
                if vd is not None and 0 <= vd < frame_height:
                    cv2.line(annotated_frame, (obj['bbox'][0], vd), (obj['bbox'][2], vd), (255, 0, 255), 1) # Magenta
        
        # Update map with current global camera pose and all known tree world positions
        # Ensure `world_T_camera` is the pose of the *current* visualized frame.
        map_view = visualizer.update_map(world_T_camera, all_tree_world_positions)

        # Display
        # cv2.imshow("Ground Mask Debug", ground_mask) # Uncomment for debugging
        cv2.imshow("Annotated Frame", annotated_frame)
        cv2.imshow("Top-Down Map", map_view)

        # Write to 1FPS output video (as per assignment deliverable 2.4.a)
        if frame_id_counter % frames_to_process_for_1fps_output == 0:
            out_video_1fps.write(annotated_frame)
        
        # Log to CSV (deliverable 2.4.c)
        if current_frame_tree_data_for_csv:
            log_results_to_csv(csv_filepath, current_frame_tree_data_for_csv, header=csv_header)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break
        
        end_time_frame = time.time()
        # print(f"Frame {frame_id_counter} processed in {end_time_frame - start_time_frame:.3f} seconds.")
            
    # --- Cleanup ---
    cap.release()
    out_video_1fps.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_id_counter} frames in total.")
    print(f"Annotated video saved to: {output_video_path}")
    print(f"Tree data saved to: {csv_filepath}")
    print("Main processing finished.")

if __name__ == '__main__':
    main()