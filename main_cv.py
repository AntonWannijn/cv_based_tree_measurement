# === Stap 0: Imports en configuratie ===
import cv2
import os
import numpy as np
from Calibratie import get_camera_calibration, extract_calibration_frames
from pose_utils import extract_and_match_features, compute_essential_matrix, recover_camera_pose
import matplotlib.pyplot as plt
# Pad naar beelden
frames_dir = r"C:\Users\Progr\dataset\images_1fps_westbound"

# Extractie van calibratieframes aan 0.5 fps (1 frame om de 2s)
extract_calibration_frames(
    video_path=r"C:\Users\Progr\Downloads\computervisie_2024\computervisie_2024\calibration.MP4",
    output_dir="dataset/calibration_frames",
    extract_fps=0.5
)

# Bereken cameramatrix en distortiecoëfficiënten
K, dist = get_camera_calibration("dataset/calibration_frames")
print("K:\n", K)
print("dist:\n", dist.ravel())

# === Stap 1: Laad alle frames ===
image_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
frames = [cv2.imread(f) for f in image_files]

# === Stap 2: (later) Detecteer bomen in elk frame ===
# - Gebruik boomdetectiemodel (YOLO of ander) om bounding boxes te verkrijgen.
# - Sla per frame de box-coördinaten op in een datastructuur of bestand.

#Meerdere labels voorzien (kani voorziet paaltje)

                        # === Stap 0: Imports en configuratie ===
                        # - Importeer alle nodige libraries: torch, torchvision, cv2, numpy, matplotlib, os
                        # - Stel pad naar dataset en annotaties in
                        # - Optioneel: gebruik argparse of config.yaml om parameters flexibel te houden

                        # === Stap 1: Datasetvoorbereiding ===
                        # - Lees annotaties in (YOLO-formaat, Pascal VOC, COCO,...)
                        # - Splits dataset in train/val/test sets
                        # - Pas data augmentatie toe: horizontale flips, kleurverandering, crop, resize
                        # - Gebruik torchvision.transforms of Albumentations (mogelijk)

                        # === Stap 2: Laad een pre-trained model (transfer learning) ===
                        # - Start van een model zoals YOLOv5, Faster R-CNN, SSD, RetinaNet
                        # - Gebruik pre-trained gewichten op COCO of Pascal VOC
                        # - Mogelijkheid: freeze de basislayers als je weinig data hebt
                        # - Optioneel: vervang alleen de laatste classificatielaag voor je eigen klassen

                        # === Stap 3: Lossfuncties en optimizer instellen ===
                        # - Combineer: box-regressieloss, classificatieloss en object confidence loss
                        # - Gebruik optimizer: Adam of SGD met momentum
                        # - Stel scheduler in voor learning rate aanpassing (mogelijk)

                        # === Stap 4: Trainingsloop opzetten ===
                        # - Itereer over epochs:
                        #     - per batch: bereken loss, backpropagation, update gewichten
                        #     - log training/val loss (via console, TensorBoard of Weights & Biases)
                        # - Voeg checkpointing toe: sla beste model op (laagste val loss)

                        # === Stap 5: Evaluatie na elke epoch ===
                        # - Bereken metrics zoals:
                        #     - mAP (mean Average Precision)
                        #     - Precision, Recall (hoge precisie is soms een overslaan maar wel altijd de juiste voorspellen, hoge recall is alles altijd oppikken)
                        #     - IOU (Intersection over Union)
                        # - Gebruik bibliotheken zoals `torchmetrics`, `pycocotools`, of `mean_average_precision`

                        # === Stap 6: Opslaan van het model ===
                        # - Sla het getrainde model op in `.pt` (PyTorch), `.h5` (Keras), of `.onnx` formaat
                        # - Optioneel: exporteer naar TensorRT of CoreML voor deployment op embedded devices

                        # === Stap 7: Inference script opzetten ===
                        # - Laad een afbeelding of video
                        # - Pas dezelfde preprocessing toe als tijdens training
                        # - Laat model voorspellingen doen
                        # - Teken bounding boxes + labels op de afbeelding
                        # - Optioneel: sla afbeeldingen op met overlay of toon ze live

                        # === Stap 8: Test op generalisatie ===
                        # - Test het model op andere seizoenen, locaties, lichtomstandigheden
                        # - Noteer performanceverlies indien van toepassing
                        # - Fine-tune eventueel op nieuwe beelden (mogelijk via frozen base en nieuwe head

# === Stap 3–5: Bereken matches, E-matrix, R & t tussen opeenvolgende frames ===
camera_poses = []  # Hierin bewaren we cumulatief R, t
R_total = np.eye(3)
t_total = np.zeros((3, 1))
camera_poses.append((R_total.copy(), t_total.copy()))

for i in range(len(frames) - 1):
    img1 = frames[i]
    img2 = frames[i+1]

    pts1, pts2, match_vis = extract_and_match_features(img1, img2)
    if len(pts1) < 8:
        print(f"❗ Te weinig matches tussen frame {i} en {i+1}, overslaan...")
        continue

    E, F, mask = compute_essential_matrix(pts1, pts2, K)
    R, t, mask_pose = recover_camera_pose(E, pts1, pts2, K)

    # Cumulatief pad opbouwen
    t_total += R_total @ t
    R_total = R @ R_total
    camera_poses.append((R_total.copy(), t_total.copy()))

    print(f"✅ Frame {i} → {i+1}: beweging berekend.")

    # (optioneel) visualiseer matches
    # cv2.imshow("Matches", match_vis)
    # cv2.waitKey(1)

# === Stap 6: (later) Koppel bomen tussen frames ===
# - Vergelijk bounding boxes over frames (IoU of centroid-afstand).
# - Wijs tracking-ID toe aan gekoppelde bomen.

# === Stap 7: (later) Trianguleer 3D-positie van gekoppelde bomen ===
# - Gebruik centerpunten van boxes + P1 = K[R|t] + triangulatePoints().
# - Zet om naar cartesisch coördinatensysteem (X/Z-plot).

# === Stap 8: Bewaar resultaten ===
# - Schrijf CSV met boom-ID, (X, Z), aantal frames, eventueel DBH.
# - Sla cameratraject op als lijst met cumulatieve posities.

# === Stap 9: Visualiseer resultaten ===
# - Plot top-down map van boomposities + cameratraject.
# - Optioneel: maak debug-annotaties op de beelden.

# === Stap 10: Exporteer video met annotaties (optioneel) ===
# - Teken boxes, boom-ID’s en schrijf nieuwe video weg aan 1 fps.
