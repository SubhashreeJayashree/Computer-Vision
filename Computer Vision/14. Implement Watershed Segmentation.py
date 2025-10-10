import cv2
import numpy as np

# Read image
img = cv2.imread("coins.jpg")

# Check if the image loaded properly
if img is None:
    raise ValueError("Error: Image not found. Please check the file path.")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal using morphological opening
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Sure foreground area using distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0 but 1
markers = markers + 1

# Mark the unknown region with 0
markers[unknown == 255] = 0

# Apply the watershed algorithm
markers = cv2.watershed(img, markers)

# Boundary marked in red
img[markers == -1] = [255, 0, 0]

# Display result
cv2.imshow("Watershed", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
