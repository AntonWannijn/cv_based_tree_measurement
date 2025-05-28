import cv2
import numpy as np
import glob
import os

# Checkerboard-afmetingen → 10 horizontaal, 6 verticaal
CHECKERBOARD = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D wereldpunten voorbereiden
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 3D punten
imgpoints = []  # 2D punten

# ✅ GEBRUIK HET NIEUWE PAD NAAR DE FRAMES
image_path = r'D:\dataset\calibration_frames\*.jpg'
images = glob.glob(image_path)

print(f"Gevonden {len(images)} beelden in {os.path.dirname(image_path)}")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Zoek checkerboard-hoeken
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Teken en toon de hoeken
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Calibration', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Kalibreer de camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("✅ Camera matrix:\n", mtx)
print("✅ Distortion coefficients:\n", dist)

# Sla resultaten op
np.savez('D:\\calibration_data_10x6.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print("✅ Calibratiegegevens opgeslagen als 'D:\\calibration_data_10x6.npz'")

# ====== EXTRA: EEN FRAME CORRIGEREN MET CALIBRATIE ======
print("\n🔧 Frame corrigeren met calibratiegegevens...")

# Frame inladen
frame_path = r'D:\dataset\images_1fps_westbound\frame_00000.jpg'
frame = cv2.imread(frame_path)

if frame is None:
    print(f"⚠️ Kon frame niet laden: {frame_path}")
else:
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Correctie toepassen (undistort)
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # ROI bijsnijden als nodig
    x, y, w, h = roi
    undistorted_cropped = undistorted[y:y+h, x:x+w]

    # Toon en sla op
    cv2.imshow('Origineel', frame)
    cv2.imshow('Gecorrigeerd', undistorted_cropped)
    cv2.imwrite('D:\\undistorted_frame.jpg', undistorted_cropped)
    print("💾 Gecorrigeerd frame opgeslagen als 'D:\\undistorted_frame.jpg'")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
