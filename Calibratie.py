import cv2
import os
import numpy as np
import glob

def extract_calibration_frames(video_path, output_dir, extract_fps=0.5):
    """
    Extraheert frames aan gegeven fps (standaard 0.5 = 1 frame per 2s) uit calibratievideo.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"❌ Fout bij openen van video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    interval = int(fps / extract_fps)

    print(f"🎥 Extractie uit: {video_path}")
    print(f"- Originele FPS: {fps:.2f}, duur: {duration:.1f} s")
    print(f"- Interval: elke {interval} frames (≈ {1/extract_fps:.1f} s)")

    saved = 0
    for frame_number in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_name = os.path.join(output_dir, f"calib_frame_{frame_number:06d}.jpg")
        cv2.imwrite(frame_name, frame)
        saved += 1

    cap.release()
    print(f"✅ {saved} calibratieframes opgeslagen in '{output_dir}'")

def get_camera_calibration(calib_images_dir, checkerboard_size=(8, 6), cache_file='camera_calibration.npz'):
    """
    Laadt cameramatrix en distortiecoëfficiënten of voert calibratie uit op basis van checkerboardbeelden.
    """
    import os
    import numpy as np
    import cv2
    import glob

    if os.path.exists(cache_file):
        print(f"📦 Camerakalibratie geladen uit {cache_file}")
        data = np.load(cache_file)
        return data['K'], data['dist']

    print(f"📐 Geen cache gevonden. Start calibratie met beelden uit: {calib_images_dir}")
    objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = sorted(glob.glob(os.path.join(calib_images_dir, "*.jpg")))
    if len(images) == 0:
        raise FileNotFoundError(f"❌ Geen beelden gevonden in {calib_images_dir}")

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    valid = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners2)
            valid += 1
        else:
            print(f"⚠️ Checkerboard niet gevonden in {fname}")

    if valid < 10:
        raise ValueError(f"❌ Te weinig geldige checkerboardbeelden gevonden ({valid}). Minimaal 10 vereist.")

    ret, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    np.savez(cache_file, K=K, dist=dist)
    print(f"✅ Calibratie voltooid. Cameramatrix en distortie opgeslagen in {cache_file}")
    return K, dist


def undistort_image(img, K, dist):
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, K, dist, None, new_K)
    return undistorted


