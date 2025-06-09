import cv2
import numpy as np
import glob
import os

# === Instellingen ===
calibration_file = r'D:\calibration_data_10x6.npz'
input_folder = r'D:\dataset\images_1fps_westbound'
output_folder = r'D:\dataset\images_undistorted'

# Zorg dat de outputmap bestaat
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Laad calibratiegegevens
calib_data = np.load(calibration_file)
mtx = calib_data['mtx']
dist = calib_data['dist']

# Pak alle .jpg-bestanden in de inputmap
image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
print(f"Gevonden {len(image_paths)} bestanden in {input_folder}")

if len(image_paths) == 0:
    print("Geen beelden gevonden om te verwerken.")
    exit()

# Verwerk elk beeld
for idx, frame_path in enumerate(image_paths):
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Kon afbeelding niet laden: {frame_path}")
        continue

    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # Correctie toepassen
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Snij bij tot geldige regio
    x, y, w_roi, h_roi = roi
    undistorted_cropped = undistorted[y:y+h_roi, x:x+w_roi]

    # Resizen naar originele resolutie (optioneel)
    undistorted_resized = cv2.resize(undistorted_cropped, (frame.shape[1], frame.shape[0]))

    # Opslaan
    filename = os.path.basename(frame_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, undistorted_resized)
    print(f"[{idx + 1}/{len(image_paths)}] ✅ Gecorrigeerd opgeslagen als: {output_path}")


