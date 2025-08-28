import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("robot.jpg")

#SIFT
gray=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
sift=cv2.SIFT_create()
kp,des=sift.detectAndCompute(gray,None)
sift_img=cv2.drawKeypoints(gray,kp,None)

#HOG
hog=cv2.HOGDescriptor()
h=hog.compute(gray)
plt.subplot(1,2,1),plt.imshow(sift_img),plt.title("SIFT Features")
plt.title("HOG Features(first 100)")
plt.axis("off")
plt.show()