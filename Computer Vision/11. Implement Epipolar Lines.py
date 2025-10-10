import cv2
import numpy as np

# Read images in grayscale
img1 = cv2.imread("left.jpg", 0)
img2 = cv2.imread("right.jpg", 0)

# Check if images loaded correctly
if img1 is None or img2 is None:
    raise ValueError("Error: One or both images not found. Check file paths.")

# Detect ORB features
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance (optional but helps in stability)
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched keypoints
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Compute Fundamental matrix
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

# Select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# Draw epilines
def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r_line, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r_line[2] / r_line[1]])
        x1, y1 = map(int, [c, -(r_line[2] + r_line[0] * c) / r_line[1]])
        cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img1, tuple(pt1.astype(int)), 5, color, -1)
        cv2.circle(img2, tuple(pt2.astype(int)), 5, color, -1)
    return img1, img2

# Compute epilines corresponding to points in right image (drawn on left image)
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)

img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Display results
cv2.imshow("Epilines Left", img5)
cv2.imshow("Epilines Right", img6)
cv2.waitKey(0)
cv2.destroyAllWindows()
