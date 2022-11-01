import cv2
import matplotlib.pyplot as plt

# GRADIENTS

# Image gradient is used to extract information from an image. 
# It is one of the fundamental building blocks in image processing and edge detection. 
# The main application of image gradient is in. 
# Many algorithms, such as Canny Edge Detection, use image gradients for detecting edges

img=cv2.imread("sudo.jpg")

plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Sudoku Image")

# X GRADIENT (SOBEL)

sobelX=cv2.Sobel(img,ddepth=cv2.CV_16S,dx=1,dy=0,ksize=5)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(sobelX),plt.title('Sobl X')
plt.xticks([]), plt.yticks([])
plt.show()

# Y GRADIENT (SOBEL)

sobelY=cv2.Sobel(img,ddepth=cv2.CV_16S,dx=0,dy=1,ksize=5)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(sobelY),plt.title('Sobl Y')
plt.xticks([]), plt.yticks([])
plt.show()

# LAPLACIAN GRADIENT

laplacian=cv2.Laplacian(img,ddepth=cv2.CV_16S)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(laplacian),plt.title('Laplacian')
plt.xticks([]), plt.yticks([])
plt.show()