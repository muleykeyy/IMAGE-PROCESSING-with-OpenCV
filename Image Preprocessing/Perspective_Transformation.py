import cv2
import numpy as np

img=cv2.imread("audi.jpg")
cv2.imshow("My Babe",img)
cv2.waitKey(0)

print(img.shape)

width=599
height=1200

pts1=np.float32([[230,1],[1,472],[540,150],[338,617]])
pts2=np.float32([[0,0],[200,height],[width,300],[width,height]])


matrix=cv2.getPerspectiveTransform(pts1,pts2)
print(matrix)

imgOutput=cv2.warpPerspective(img,matrix,(width,height))
cv2.imshow("Output",imgOutput)
cv2.waitKey(0)
