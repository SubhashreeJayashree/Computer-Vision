import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("robot.jpg")
cv2.imwrite("compressed.jpg",img,[cv2.IMWRITE_JPEG_QUALITY,50])
cv2.imwrite("compressed.png",img,[cv2.IMWRITE_PNG_COMPRESSION,9])
cv2.imwrite("compressed.webp",img,[cv2.IMWRITE_WEBP_QUALITY,50])