import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGUREERBARE PARAMETERS ===
frames_dir = r"D:\dataset\images_1fps_westbound"
output_dir = os.path.join(frames_dir, "sift_matches_filtered")
ratio_thresh = 0.70       # Strenger dan standaard 0.75
top_N = 100              # Alleen de 100 beste matches tonen
distance_thresh = 200    # Maximaal toegestane afstand

# Camera matrix (vul hier jouw echte kalibratiewaarden in)
K = np.array([[1.41441241e+03, 0, 1.33590825e+03],
              [0, 1.42677591e+03, 7.99013747e+02],
              [0, 0, 1]])

# Maak outputmap aan
os.makedirs(output_dir, exist_ok=True)

# Verzamel en sorteer de .jpg-bestanden
image_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])

# Initialiseer SIFT en matcher
sift = cv.SIFT_create()
bf = cv.BFMatcher()

# Voor padberekening
trajectory = [np.array([0, 0, 0])]
current_R = np.eye(3)
current_t = np.zeros((3, 1))
last_t = np.zeros((3, 1))  # fallback translatie

for i in range(len(image_files) - 1):
    img1 = cv.imread(image_files[i])
    img2 = cv.imread(image_files[i + 1])

    if img1 is None or img2 is None:
        print(f"Frame {i} of {i + 1} niet gevonden, overslaan.")
        continue

    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Detecteer keypoints en descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print(f"Geen descriptors gevonden in frame {i} of {i+1}, overslaan.")
        continue

    # Match descriptors met KNN
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio filtering
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    # Sorteer op afstand (beste eerst)
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # Filter op afstandsdrempel
    good_matches = [m for m in good_matches if m.distance < distance_thresh]

    # Pak alleen top N
    good_matches = good_matches[:top_N]

    print(f"Frame {i} ↔ {i+1}: {len(good_matches)} goede matches geselecteerd.")

    # === Padberekening toevoegen ===
    if len(good_matches) >= 8:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, prob=0.999, threshold=1.0)

        if E is not None:
            _, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, K)
            last_t = t.copy()
            current_t += current_R @ t
            current_R = R @ current_R
            print(f"Frame {i}-{i+1}: positie {current_t.flatten()}")
        else:
            print(f"Frame {i}-{i+1}: geen essentiële matrix, fallback vorige beweging.")
            current_t += current_R @ last_t
    else:
        print(f"Frame {i}-{i+1}: te weinig matches, fallback vorige beweging.")
        current_t += current_R @ last_t

    trajectory.append(current_t.flatten())

    # === Matches tekenen en opslaan ===
    matched_img = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    output_path = os.path.join(output_dir, f'matches_{i:04d}_{i+1:04d}.jpg')
    cv.imwrite(output_path, matched_img)
    print(f"Opgeslagen: {output_path}")

print("Alle gefilterde match-afbeeldingen zijn opgeslagen.")

# === Traject plotten ===
trajectory = np.array(trajectory)
plt.figure(figsize=(8, 8))
plt.plot(trajectory[:, 0], trajectory[:, 2], marker='o')  # X vs Z (horizontaal pad)
plt.xlabel('X (links-rechts)')
plt.ylabel('Z (voor-achter)')
plt.title('Geschat pad van camera/brommer')
plt.grid()
plt.axis('equal')
plt.show()
