# visualizer.py
import cv2
import numpy as np

class Visualizer:
    def __init__(self, map_size=(800, 800), map_scale_pixels_per_meter=20):
        self.map_image = np.ones((map_size[0], map_size[1], 3), dtype=np.uint8) * 240 # Light gray map
        self.map_scale = map_scale_pixels_per_meter
        self.map_origin_px = (map_size[1] // 2, map_size[0] - 50) # X, Y for map origin (e.g. bottom center)
        self.camera_path_points = [] # Store (x,z) world coordinates

    def draw_on_frame(self, frame, tracked_objects, dbh_results):
        """
        Draws bounding boxes, track IDs, and DBH on the frame.
        tracked_objects: list from tracker {'bbox': ..., 'track_id': ...}
        dbh_results: dict {track_id: dbh_cm}
        """
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            track_id = obj['track_id']
            
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue for bbox
            
            label = f"ID: {track_id}"
            if track_id in dbh_results and dbh_results[track_id] > 0:
                label += f" DBH: {dbh_results[track_id]:.1f}cm"
            
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        return frame

    def update_map(self, camera_world_pose_mat, tree_world_positions):
        """
        Updates the top-down map with camera trajectory and tree positions.
        camera_world_pose_mat: 4x4 transformation matrix of camera in world frame.
        tree_world_positions: dict {track_id: (X, Y, Z world_coords)}
        """
        # Current camera position (assuming world X, Z are ground plane)
        cam_x_world = camera_world_pose_mat[0, 3]
        cam_z_world = camera_world_pose_mat[2, 3]
        self.camera_path_points.append((cam_x_world, cam_z_world))

        # Clear previous map drawing elements (or draw fresh)
        self.map_image = np.ones_like(self.map_image) * 240

        # Draw camera trajectory
        if len(self.camera_path_points) > 1:
            for i in range(len(self.camera_path_points) - 1):
                pt1_world = self.camera_path_points[i]
                pt2_world = self.camera_path_points[i+1]
                cv2.line(self.map_image, self._world_to_map_px(pt1_world), self._world_to_map_px(pt2_world), (0,0,255), 1) # Red path

        # Draw current camera position
        cv2.circle(self.map_image, self._world_to_map_px((cam_x_world, cam_z_world)), 5, (255,0,0), -1) # Blue dot for camera

        # Draw trees
        for track_id, world_coords in tree_world_positions.items():
            tree_x_world, _, tree_z_world = world_coords # Using X, Z for map
            map_pt = self._world_to_map_px((tree_x_world, tree_z_world))
            cv2.circle(self.map_image, map_pt, 4, (0, 128, 0), -1) # Dark Green for trees
            cv2.putText(self.map_image, str(track_id), (map_pt[0]+5, map_pt[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            
        return self.map_image

    def _world_to_map_px(self, world_xz_coords):
        """Converts world (X,Z) ground plane coordinates to map pixel coordinates."""
        world_x, world_z = world_xz_coords
        map_px_x = self.map_origin_px[0] + int(world_x * self.map_scale)
        # Z is typically forward, map Y is often down. If world Z increases "away", map Y increases "down".
        map_px_y = self.map_origin_px[1] - int(world_z * self.map_scale) # Z positive = further away = higher on map (up from origin) if origin is bottom
        return (map_px_x, map_px_y)

if __name__ == '__main__':
    vis = Visualizer()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Example tracked objects and DBH
    tracked = [{'bbox': (50,50,150,300), 'track_id': 1}, {'bbox': (200,60,280,320), 'track_id': 2}]
    dbhs = {1: 25.5, 2: 30.1}
    
    annotated_f = vis.draw_on_frame(dummy_frame.copy(), tracked, dbhs)
    # cv2.imshow("Annotated Frame", annotated_f)
    
    # Example map update
    # Dummy camera poses (identity, then moved 1m in X, 2m in Z)
    cam_pose1 = np.eye(4) 
    cam_pose2 = np.array([[1,0,0,1.0], [0,1,0,0], [0,0,1,2.0], [0,0,0,1]])
    
    trees_world = {1: (0.5, 0, 3.0), 2: (-0.5, 0, 2.5)} # (X,Y,Z) Y is up, so ignored for 2D map

    map_img = vis.update_map(cam_pose1, {}) # Initial camera pos
    map_img = vis.update_map(cam_pose2, trees_world) # Moved camera, trees visible
    
    # cv2.imshow("Map View", map_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("Visualizer example run (images would show if waitKey was not commented).")