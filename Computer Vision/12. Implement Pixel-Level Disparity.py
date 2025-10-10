import cv2
import numpy as np

# Load stereo pair (grayscale)
imgL = cv2.imread("left.jpg", 0)
imgR = cv2.imread("right.jpg", 0)

# Check if images are loaded correctly
if imgL is None or imgR is None:
    raise ValueError("Error: One or both images not found. Please check file paths.")

# Create StereoBM object
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)

# Compute disparity map
disparity = stereo.compute(imgL, imgR)

# Normalize for display (convert disparity to 8-bit image)
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# Show the disparity map
cv2.imshow("Disparity", disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
