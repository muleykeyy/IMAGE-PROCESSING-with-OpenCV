import cv2
import matplotlib.pyplot as plt

img=cv2.imread("view2.jpg",0)


plt.figure()
plt.imshow(img,cmap="gray")
plt.axis("off")
plt.show()

# THRESHOLD
_,thresh_img_white=cv2.threshold(img,thresh=60,maxval=255,type=cv2.THRESH_BINARY)
# cv2.TRESH_BINARY make white between 60-255
_,thresh_img_black=cv2.threshold(img,thresh=60,maxval=255,type=cv2.THRESH_BINARY_INV)
# cv2.TRESH_BINARY_INV make black between 60-255

# WHITE
plt.figure()
plt.imshow(thresh_img_white,cmap="gray")
plt.axis("off")
plt.show()

# BLACK
plt.figure()
plt.imshow(thresh_img_black,cmap="gray")
plt.axis("off")
plt.show()

# ADAPTIVE THRESHOLD

thresh_img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,5)
plt.figure()
plt.imshow(thresh_img,cmap="gray")
plt.axis("off")
plt.show()