import cv2
import matplotlib.pyplot as plt

img1=cv2.imread("view1.jpg") # OpenCV import images in BGR format. We need to convert it to RGB
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2=cv2.imread("view2.jpg")
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)


plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

# Check the shape of images:

print(img1.shape)
print(img2.shape)

# They have different shape. We should make them equal.

img1=cv2.resize(img1,(600,600))
img2=cv2.resize(img2,(600,600))
print("Image 1: ",img1.shape,"Image 2: ",img2.shape)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

# Blending Image= alpha x img1 + beta x img2

blended=cv2.addWeighted(src1=img1,alpha=0.5,src2=img2,beta=0.7,gamma=0)
plt.figure()
plt.imshow(blended)
