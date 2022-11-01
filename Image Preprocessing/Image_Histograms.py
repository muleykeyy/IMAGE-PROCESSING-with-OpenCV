import cv2
import matplotlib.pyplot as plt
import numpy as np

# HISTOGRAM

# A histogram represents the distribution of pixel intensities (whether color or grayscale) in an image. 
# It can be visualized as a graph (or plot) that gives a high-level intuition of the intensity (pixel value) distribution. 
# We are going to assume a RGB color space in this example, so these pixel values will be in the range of 0 to 255.

img=cv2.imread("colors.jpeg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Color Image")

print(img.shape)

img_hist=cv2.calcHist([img],channels=[0],mask=None,histSize=[256],ranges=[0,256])
print(img_hist.shape)
plt.figure()
plt.plot(img_hist)

color=["b","r","g"]
plt.figure()

for i,c in enumerate(color):
    hist=cv2.calcHist([img],channels=[i],mask=None,histSize=[256],ranges=[0,256])
    plt.plot(hist,color=c)
    
# MASKING

car=cv2.imread("audi.jpg")
car=cv2.cvtColor(car,cv2.COLOR_BGR2RGB)


plt.figure()
plt.imshow(car, cmap="gray")
plt.title("Car")
print(car.shape)

# Create mask
mask=np.zeros(car.shape[:2],np.uint8)
plt.figure()
plt.imshow(mask, cmap="gray")
plt.axis("off")
plt.title("MASK")

mask[200:400,200:800]=255
plt.figure()
plt.imshow(mask, cmap="gray")

# Add image

masked_img=cv2.bitwise_and(car,car,mask=mask)
plt.figure()
plt.imshow(masked_img, cmap="gray")
plt.axis("off")
plt.title("Masked Image")

# Apply Histogram to Masked Image

masked_img_hist=cv2.calcHist([car],channels=[0],mask=mask,histSize=[256],ranges=[0,256])
plt.figure()
plt.plot(masked_img_hist)
plt.title("Masked Image Histogram")


# HISTOGRAM EQUALIZATION

img=cv2.imread("hist_equ.jpg",0)
plt.figure()
plt.imshow(img,cmap="gray")
plt.axis("off")
plt.title("Orginal Img")
plt.show()

img_hist=cv2.calcHist([img],channels=[0],mask=None,histSize=[256],ranges=[0,256])

plt.figure()
plt.plot(img_hist)

hist_equ=cv2.equalizeHist(img)

plt.subplot(121),plt.imshow(img,cmap="gray"),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(hist_equ,cmap="gray"),plt.title('Equalization')
plt.xticks([]), plt.yticks([])
plt.show()

hist_equ_cal=cv2.calcHist([hist_equ],channels=[0],mask=None,histSize=[256],ranges=[0,256])

plt.figure()
plt.plot(hist_equ_cal)
plt.title("After Equ")
