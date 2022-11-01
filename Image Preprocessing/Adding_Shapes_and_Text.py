import cv2
import numpy as np

# Create an Image

img=np.zeros((512,512,3),np.uint8)
cv2.imshow("Black",img)
cv2.waitKey(0)

# ADD LINE

# cv2.line(image, start point, end point,color,width of line)
cv2.line(img,(0,0),(512,512),(0,255,0),3)
# In OpenCV RGB -> BGR 

#   R in OpenCV: (0,0,255)
#   G in OpenCV: (0,255,0)
#   B in OpenCV: (255,0,0)
cv2.imshow("LINE",img)
cv2.waitKey(0)

# RECTANGLE

# cv2.rectangle(image, start point, end point,color,width of lines)
cv2.rectangle(img,(0,0),(256,256),(255,0,0),5)
cv2.imshow("RECTANGLE",img)
cv2.waitKey(0)

cv2.rectangle(img,(0,0),(256,256),(255,0,0),cv2.FILLED)
cv2.imshow("RECTANGLE FILLED",img)
cv2.waitKey(0)

# CIRCLE

# cv2.circle(image,center,radius,color)
cv2.circle(img,(300,300),45,(0,0,255))
cv2.imshow("CIRCLE",img)
cv2.waitKey(0)


cv2.circle(img,(300,300),45,(0,0,255),cv2.FILLED)
cv2.imshow("CIRCLE FILLED",img)
cv2.waitKey(0)

# TEXT

# cv2.putText(image, "TEXT",(starting point), font ,width of text ,(color of text))
cv2.putText(img, "Muleyke",(350,350),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))
cv2.imshow("TEXT",img)
cv2.waitKey(0)
