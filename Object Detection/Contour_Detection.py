import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread("contour.jpg",0)

plt.figure()
plt.imshow(img,cmap="gray")
plt.axis("off")
plt.show()

contours,hierarcy=cv2.findContours(img,mode=cv2.RETR_CCOMP,method=cv2.CHAIN_APPROX_SIMPLE)

external_contours=np.zeros(img.shape)
internal_contours=np.zeros(img.shape)

for i in range(len(contours)):
    
    # External
    if hierarcy[0][i][3]== -1:
        cv2.drawContours(external_contours,contours,i,255,-1)
    else: # Internal
        cv2.drawContours(internal_contours,contours,i,255,-1)
        
plt.figure()
plt.imshow(external_contours,cmap="gray")
plt.axis("off")
plt.title("External")
plt.show()

plt.figure()
plt.imshow(internal_contours,cmap="gray")
plt.axis("off")
plt.title("Internal")
plt.show()        
