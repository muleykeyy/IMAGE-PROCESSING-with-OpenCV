import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("sudo.jpg",0)
img=np.float32(img)
print(img.shape)

plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Original")
plt.show()

# HARRIS CORNER DETECTOR

dst=cv2.cornerHarris(img,blockSize=5,ksize=3,k=1)
dst=cv2.dilate(dst,None)
plt.figure()
plt.imshow(dst, cmap="gray")
plt.axis("off")
plt.title("Harris Corner")
plt.show()

# SHI - TOMASI DETECTION

img=cv2.imread("sudo.jpg",0)
img=np.float32(img)

corners=cv2.goodFeaturesToTrack(img,50,0.01,10) # 50 -> Number of Corners.
corners=np.uint64(corners)

for i in corners:
    x,y=i.ravel()
    cv2.circle(img,(x,y),3,(125,125,125),cv2.FILLED)
plt.imshow(img)
plt.axis("off")