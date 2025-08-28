import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("robot.jpg")
resized=cv2.resize(img,(200,200))
flipped=cv2.flip(img,1)
rotated=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
titles=["Original","Resized","Flipped","Rotated"]
images=[img,resized,flipped,rotated]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i]),plt.title(titles[i])
    plt.axis("off")
plt.show()