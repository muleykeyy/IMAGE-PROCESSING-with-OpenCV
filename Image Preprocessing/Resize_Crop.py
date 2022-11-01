# RESIZE & CROP IMAGE
import cv2

img=cv2.imread("lenna.png")
print("Size of Image:",img.shape)
cv2.imshow("ORJ Image",img)
cv2.waitKey(0)

# RESIZE
imgResized=cv2.resize(img,(800,800))
print("New shape of Image: ",imgResized.shape)
cv2.imshow("RESIZED Image",imgResized)
cv2.waitKey(0)

# CROP
imgCropped=img[:200,:300]
print("Crpped shape of Image: ",imgCropped.shape)
cv2.imshow("Cropped Image",imgCropped)
cv2.waitKey(0)