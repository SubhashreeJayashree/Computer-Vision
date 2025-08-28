import cv2
import matplotlib.pyplot as plt
img=cv2.imread("robot.jpg")
blur=cv2.blur(img,(5,5))
gaussian=cv2.GaussianBlur(img,(5,5),0)
median=cv2.medianBlur(img,5)
titles=["Original","Blur","Gaussian","Median"]
images=[img,blur,gaussian,median]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i]),plt.title(titles[i])
    plt.axis("off")
plt.show()