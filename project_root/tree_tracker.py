# tree_tracker.py
import numpy as np

def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    Boxes are (x1, y1, x2, y2).
    """
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    if intersection_area == 0:
        return 0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0:
        return 0
        
    return intersection_area / union_area

class BasicTracker:
    def __init__(self, iou_threshold=0.3, max_staleness=5):
        self.next_track_id = 0
        self.tracks = {}  # {track_id: {'bbox': bbox, 'staleness': 0, 'frames_seen':0}}
        self.iou_threshold = iou_threshold
        self.max_staleness = max_staleness # Frames to keep a track if not matched

    def update(self, detections):
        """
        Updates tracks with new detections.
        Detections: list of {'bbox': (x1, y1, x2, y2), ...}
        Returns: list of {'bbox': bbox, 'track_id': id, ... (other info from detection)}
        """
        current_frame_tracked_objects = []

        # Try to match detections to existing tracks
        matched_track_ids = set()
        temp_detections = list(detections) # Work with a copy

        for track_id, track_data in list(self.tracks.items()):
            best_match_iou = 0
            best_match_idx = -1
            
            for i, det in enumerate(temp_detections):
                iou = calculate_iou(track_data['bbox'], det['bbox'])
                if iou > self.iou_threshold and iou > best_match_iou:
                    best_match_iou = iou
                    best_match_idx = i
            
            if best_match_idx != -1: # Found a match
                matched_det = temp_detections.pop(best_match_idx)
                self.tracks[track_id]['bbox'] = matched_det['bbox']
                self.tracks[track_id]['staleness'] = 0
                self.tracks[track_id]['frames_seen'] +=1
                
                tracked_obj = matched_det.copy()
                tracked_obj['track_id'] = track_id
                current_frame_tracked_objects.append(tracked_obj)
                matched_track_ids.add(track_id)
            else: # No match for this track
                self.tracks[track_id]['staleness'] += 1

        # Remove stale tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['staleness'] > self.max_staleness:
                del self.tracks[track_id]

        # Add new detections as new tracks
        for det in temp_detections:
            self.tracks[self.next_track_id] = {'bbox': det['bbox'], 'staleness': 0, 'frames_seen':1}
            tracked_obj = det.copy()
            tracked_obj['track_id'] = self.next_track_id
            current_frame_tracked_objects.append(tracked_obj)
            self.next_track_id += 1
            
        return current_frame_tracked_objects

if __name__ == '__main__':
    tracker = BasicTracker()
    # Example: Detections in frame 1
    detections_f1 = [{'bbox': (10, 10, 50, 50), 'confidence': 0.9}]
    tracked_f1 = tracker.update(detections_f1)
    print("Frame 1 tracks:", tracked_f1)

    # Example: Detections in frame 2
    detections_f2 = [{'bbox': (12, 12, 52, 52), 'confidence': 0.88}, {'bbox': (100, 100, 150, 150), 'confidence':0.95}]
    tracked_f2 = tracker.update(detections_f2)
    print("Frame 2 tracks:", tracked_f2)

    detections_f3 = [{'bbox': (105, 105, 155, 155), 'confidence': 0.95}]
    tracked_f3 = tracker.update(detections_f3)
    print("Frame 3 tracks:", tracked_f3)
    for _ in range(6): # Simulate 6 frames with no match for track 0
        tracked_f_empty = tracker.update([{'bbox': (105, 105, 155, 155), 'confidence': 0.95}]) # only track 1 seen
    print("After staleness:", tracker.tracks)