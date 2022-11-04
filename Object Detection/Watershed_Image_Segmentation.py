import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread("coins.jpg")

plt.figure()
plt.imshow(img)
plt.title("Original")
plt.axis("off")
plt.show()

# Blur Image

img_blur=cv2.medianBlur(img,13)
plt.figure()
plt.imshow(img_blur)
plt.title("Blur")
plt.axis("off")
plt.show()

# GRAY SCALE

img_gray=cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(img_gray,cmap="gray")
plt.title("Gray Scale")
plt.axis("off")
plt.show()

# BINARY THRESHOLD

ret, img_thresh=cv2.threshold(img_gray,75,255,cv2.THRESH_BINARY)
plt.figure()
plt.imshow(img_thresh,cmap="gray")
plt.title("Blur")
plt.axis("off")


# CONTOUR
contours,hierarcy=cv2.findContours(img_thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    
    if hierarcy[0][i][3] == -1:
        cv2.drawContours(img,contours,i,(0,255,0),10)
        
plt.figure()
plt.imshow(img,cmap="gray")
plt.title("Blur")
plt.axis("off")
        
# OPENING

kernel=np.ones((3,3),np.uint8)
opening=cv2.morphologyEx(img_thresh,cv2.MORPH_OPEN,kernel,iterations=2)

plt.figure()
plt.imshow(opening,cmap="gray")
plt.title("Opening")
plt.axis("off")


# DISTANCE between OBJECTS
dist_transform=cv2.distanceTransform(opening,cv2.DIST_L2,5)

plt.figure()
plt.imshow(dist_transform,cmap="gray")
plt.title("Distance")
plt.axis("off")

# 

ret, sure_foreground=cv2.threshold(dist_transform,0.4*np.max(dist_transform),255,0)

plt.figure()
plt.imshow(sure_foreground,cmap="gray")
plt.title("Small")
plt.axis("off")

#
sure_background=cv2.dilate(opening,kernel,iterations=1)
sure_foreground=np.uint8(sure_foreground)
unknown=cv2.subtract(sure_background,sure_foreground)

plt.figure()
plt.imshow(unknown,cmap="gray")
plt.title("Unknown")
plt.axis("off")

# CONNECTION

ret,marker=cv2.connectedComponents(sure_foreground)
marker=marker+1
marker[unknown==255]=0

plt.figure()
plt.imshow(marker,cmap="gray")
plt.title("Connection")
plt.axis("off")

# WATERSHED for IMAGE SEGMENTATION

marker=cv2.watershed(img,marker)

plt.figure()
plt.imshow(marker,cmap="gray")
plt.title("Watershed")
plt.axis("off")



# Contour after Wateshed

contours,hierarcy=cv2.findContours(marker.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    
    if hierarcy[0][i][3] == -1:
        cv2.drawContours(img,contours,i,(255,0,0),10)
        
plt.figure()
plt.imshow(img,cmap="gray")
plt.title("Blur")
plt.axis("off")
