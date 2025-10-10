import cv2
import numpy as np

# Load stereo images (grayscale)
imgL = cv2.imread("left.jpg", 0)
imgR = cv2.imread("right.jpg", 0)

# Check if images loaded correctly
if imgL is None or imgR is None:
    raise ValueError("Error: One or both images not found. Please check file paths.")

# Compute disparity map
stereo = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=15)
disparity = stereo.compute(imgL, imgR)

# Convert disparity to float32 for 3D reprojection
disparity = np.float32(disparity) / 16.0  # StereoBM divides by 16 internally

# Reproject to 3D space
h, w = imgL.shape
f = 0.8 * w  # approximate focal length
Q = np.float32([
    [1, 0, 0, -w / 2],
    [0, -1, 0, h / 2],
    [0, 0, 0, -f],
    [0, 0, 1, 0]
])

# Reproject image to 3D
points_3D = cv2.reprojectImageTo3D(disparity, Q)

print("3D Points shape:", points_3D.shape)
