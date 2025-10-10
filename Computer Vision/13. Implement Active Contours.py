import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour

# Load the image in grayscale
img = cv2.imread("object.jpg", 0)

# Check if image was loaded
if img is None:
    raise ValueError("Error: Image not found. Please check the file path.")

# Threshold the image (optional preprocessing)
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Create initial contour (a circle around the object)
s = np.linspace(0, 2 * np.pi, 400)
x = 100 + 100 * np.cos(s)
y = 100 + 100 * np.sin(s)
init = np.array([x, y]).T.astype(np.float32)

# Apply Gaussian blur before active contour
blurred_img = cv2.GaussianBlur(img, (9, 9), 0)

# Perform active contour (snake)
snake = active_contour(
    blurred_img,
    init,
    alpha=0.015,   # elasticity (higher = smoother)
    beta=10,       # rigidity (higher = stiffer)
    gamma=0.001    # step size
)

# Display result
plt.imshow(img, cmap='gray')
plt.plot(init[:, 0], init[:, 1], '--r', label='Initial contour')
plt.plot(snake[:, 0], snake[:, 1], '-b', label='Final contour')
plt.legend()
plt.show()
