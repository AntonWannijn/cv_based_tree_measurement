import cv2
import os
import numpy as np
import csv
from ultralytics import YOLO
from Calibratie import get_camera_calibration, extract_calibration_frames
from pose_utils import extract_and_match_features, compute_essential_matrix, recover_camera_pose
import matplotlib.pyplot as plt


# === Stap 0: Kalibratieframes extraheren & camera kalibreren ===
print("=== Stap 0: Calibratieframes extraheren & inladen ===")
calib_video = r"C:\Users\Progr\Downloads\computervisie_2024\computervisie_2024\calibration.MP4"
calib_dir   = "dataset/calibration_frames"
extract_calibration_frames(video_path=calib_video, output_dir=calib_dir, extract_fps=0.5)
files = os.listdir(calib_dir)
print(f"  • Aantal calibratiefotos in '{calib_dir}': {len(files)}")
K, dist = get_camera_calibration(calib_dir)
print("  • Camera matrix K:\n", K)
print("  • Distortion coeffs:\n", dist.ravel())

# # === Stap 1: Video inladen en exact 1 fps subsamplen ===
# Pad naar beelden
frames_dir = r"C:\Users\Progr\dataset\images_1fps_westbound"
image_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
frames = [cv2.imread(f) for f in image_files]
timestamps = list(range(len(frames)))

# === Stap 2: YOLO-detectie op 1 fps-frames ===
model = YOLO("best.pt")
detected_boxes = []
for i, frame in enumerate(frames):
    res = model(frame, imgsz=640)[0]
    boxes = []
    for box in res.boxes:
        conf = float(box.conf[0])
        if conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            boxes.append((x1, y1, x2, y2))
    detected_boxes.append(boxes)
    print(f"  • Frame {i} (@{timestamps[i]}s): {len(boxes)} detecties")
assert any(len(b)>0 for b in detected_boxes), "Geen detecties gevonden op eender welke frame."


# === Stap 3: Cameraposes schatten ===
output_dir = "match_vis"
os.makedirs(output_dir, exist_ok=True)

camera_poses = []
R_tot = np.eye(3); t_tot = np.zeros((3,1))
camera_poses.append((R_tot.copy(), t_tot.copy()))
for i in range(len(frames) - 1):
    pts1, pts2, _ = extract_and_match_features(frames[i], frames[i+1])
    print(f"  • Frame-pair {i}->{i+1}: gevonden matches={len(pts1)}")
    img1 = frames[i]
    img2 = frames[i + 1]

    pts1, pts2, match_vis = extract_and_match_features(img1, img2)
    n_matches = len(pts1)

    filename = os.path.join(output_dir, f"match_{i:03d}_to_{i+1:03d}_{n_matches}_matches.jpg")
    cv2.imwrite(filename, match_vis)
    if len(pts1) < 8:
        camera_poses.append((R_tot.copy(), t_tot.copy()))
        print("    – Overslaan (te weinig matches).")
        continue
    E, _, _ = compute_essential_matrix(pts1, pts2, K)
    R, t, _ = recover_camera_pose(E, pts1, pts2, K)
    t_tot += R_tot @ t
    R_tot = R @ R_tot
    camera_poses.append((R_tot.copy(), t_tot.copy()))
    print(f"    – Pose geschat: t_tot={t_tot.ravel()}")


# with open("camera_poses.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Frame"] + [f"R{i}" for i in range(9)] + ["t_X", "t_Y", "t_Z"])
#     for i, (R, t) in enumerate(camera_poses):
#         R_flat = R.flatten()  # 3x3 → 9 waarden
#         row = [i] + list(R_flat) + [t[0,0], t[1,0], t[2,0]]
#         writer.writerow(row)


# assert len(camera_poses) == len(frames), "Aantal camera_poses komt niet overeen met frames."

# camera_poses = []
# with open("camera_poses.csv", "r") as f:
#     reader = csv.reader(f)
#     header = next(reader)  # skip header
#     for row in reader:
#         R_vals = list(map(float, row[1:10]))  # R0..R8
#         t_vals = list(map(float, row[10:13])) # t_X..t_Z
#         R = np.array(R_vals).reshape(3, 3)
#         t = np.array(t_vals).reshape(3, 1)
#         camera_poses.append((R, t))

traj = np.array([t[1].ravel() for t in camera_poses])
plt.plot(traj[:,0], traj[:,2], 'b-o')
plt.title("Top-down cameratraject")
plt.xlabel("X"); plt.ylabel("Z"); plt.grid(); plt.axis("equal")
plt.show()

# === Stap 4: Tracking + annotatie ===
# === Hulpfuncties ===
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def trianguleer_meerdere_frames(dets, camera_poses, K):
    if len(dets) < 2:
        return None
    A = []
    for fid, (u, v) in dets:
        R, t = camera_poses[fid]
        P = K @ np.hstack((R, t))
        row1 = u * P[2, :] - P[0, :]
        row2 = v * P[2, :] - P[1, :]
        A.append(row1)
        A.append(row2)
    A = np.stack(A)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    Xh /= Xh[3]
    return Xh[:3]

def project_point(X, P):
    Xh = np.append(X, 1)
    xh = P @ Xh
    return (xh[0] / xh[2], xh[1] / xh[2])

# === Projectiematrices opbouwen ===
proj_mats = [K @ np.hstack((R, t)) for R, t in camera_poses]

# === Trackingparameters ===
next_id = 0
tracks = {}
IOU_THRESH = 0.3
out = cv2.VideoWriter("tracked_result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 1.0,
                      (frames[0].shape[1], frames[0].shape[0]))

# === Tracking Loop ===
for fid, frame in enumerate(frames):
    boxes = detected_boxes[fid]
    used = set()

    # 1) Update bestaande tracks
    for tid, tr in list(tracks.items()):
        if tr['last_frame'] != fid - 1:
            continue

        matched = False
        match_idx = None

        # A) Projectie-gebaseerde matching
        if tr['X3d'] is not None:
            uv_pred = project_point(tr['X3d'], proj_mats[fid])
            dists = []
            thresholds = []
            for b in boxes:
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                dist = np.linalg.norm(np.array(uv_pred) - np.array((cx, cy)))
                box_height = b[3] - b[1]
                adaptive_thresh = max(40, 0.3 * box_height)
                dists.append(dist)
                thresholds.append(adaptive_thresh)

            if dists:
                j = int(np.argmin(dists))
                if dists[j] < thresholds[j]:
                    matched = True
                    match_idx = j

        # B) IoU-gebaseerde fallback
        elif len(tr['dets']) < 2:
            ious = [compute_iou(tr['box'], b) for b in boxes]
            if ious:
                j = int(np.argmax(ious))
                if ious[j] > IOU_THRESH:
                    matched = True
                    match_idx = j

        # C) Bij match → update track
        if matched and match_idx not in used:
            box = boxes[match_idx]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            tr['box'] = box
            tr['dets'].append((fid, (cx, cy)))
            tr['last_frame'] = fid
            used.add(match_idx)

            # Trianguleer zodra er 2+ waarnemingen zijn
            if len(tr['dets']) >= 2:
                tr['X3d'] = trianguleer_meerdere_frames(tr['dets'], camera_poses, K)

    # 2) Nieuwe tracks starten
    for i, box in enumerate(boxes):
        if i not in used:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            tracks[next_id] = {
                'box': box,
                'dets': [(fid, (cx, cy))],
                'X3d': None,
                'last_frame': fid
            }
            used.add(i)
            next_id += 1

    # 3) Annotatie
    vis = frame.copy()
    for tid, tr in tracks.items():
        if tr['last_frame'] == fid:
            x1, y1, x2, y2 = map(int, tr['box'])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"ID {tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out.write(vis)
    print(f"Frame {fid}: {len(used)} matches verwerkt")

out.release()

# === Stap 5: Reprojection error evaluatie ===

def reproj_error(P, X, uv):
    Xh = np.append(X, 1)
    xh = P @ Xh
    xh /= xh[2]
    return np.linalg.norm(np.array(uv) - xh[:2])

proj_mats = [K @ np.hstack((R, t)) for R, t in camera_poses]

errors = []
for tid, tr in tracks.items():
    if tr['X3d'] is None:
        continue
    for fid, uv in tr['dets']:
        err = reproj_error(proj_mats[fid], tr['X3d'], uv)
        errors.append(err)

errors = np.array(errors)
print(f"📏 Reprojection error (px):")
print(f"  Gemiddeld: {np.mean(errors):.2f}")
print(f"  Mediaan : {np.median(errors):.2f}")
print(f"  Maximaal: {np.max(errors):.2f}")

import matplotlib.pyplot as plt

plt.hist(errors, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram van reprojection error")
plt.xlabel("Reprojection error (px)")
plt.ylabel("Aantal waarnemingen")
plt.grid()
plt.show()


# # === Stap 5: Triangulatie per track ===
# print("\n=== Stap 5: Triangulatie ===")
# tree_positions = {}
# for tid, tr in tracks.items():
#     X, _ = trianguleer_met_error(tr['detections'], camera_poses, K)
#     if X is not None:
#         tree_positions[tid] = X
#         print(f"  • Track {tid}: 3D pos = {X}")
#     else:
#         print(f"  • Track {tid}: onvoldoende waarnemingen")

# assert tree_positions, "Geen boomposities getrianguleerd."

# # === Stap 6: CSV-export ===
# print("\n=== Stap 6: CSV-export ===")
# with open('boom_mapping.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['Tree_ID','X','Y','Z'])
#     for tid, X in tree_positions.items():
#         writer.writerow([tid, *X])
# print("  • CSV 'boom_mapping.csv' opgeslagen")

# # === Stap 7: Top-down visualisatie ===
# print("\n=== Stap 7: Visualisatie ===")
# plt.figure(figsize=(8,6))
# for tid, X in tree_positions.items():
#     x, y, z = X
#     plt.scatter(x, z)
#     plt.text(x, z, str(tid), fontsize=9)
# plt.title("Top-down kaart van boomposities")
# plt.xlabel("X (m)"); plt.ylabel("Z (m)")
# plt.grid(); plt.axis('equal')
# plt.show()
