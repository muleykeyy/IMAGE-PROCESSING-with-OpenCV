import cv2
import numpy as np


img=cv2.imread("lenna.png")
cv2.imshow("Orginal",img)
cv2.waitKey(0)

# COMBINE HORIZONTAL

hor=np.hstack((img,img)) # Images are ARRAY. So, to combine them, use numpy
cv2.imshow("Horizontal",hor)
cv2.waitKey(0)

# COMBINE VERTICAL

ver=np.vstack((img,img))
cv2.imshow("Vertical",ver)
cv2.waitKey(0)