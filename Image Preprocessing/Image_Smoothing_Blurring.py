import cv2
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

img=cv2.imread("style3.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img)
plt.axis("off")
plt.title("ORGINAL")
plt.show()

# MEAN BLURRING

# This is done by convolving an image with a normalized box filter. 
# It simply takes the average of all the pixels under the kernel area and replaces the central element.

dst2=cv2.blur(img,ksize=(15,15))

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst2),plt.title('Mean Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

# GAUSSIAN BLURRING

# In this method, instead of a box filter, a Gaussian kernel is used. 
# It is done with the function, cv.GaussianBlur(). 
# We should specify the width and height of the kernel which should be positive and odd. 
# We also should specify the standard deviation in the X and Y directions, sigmaX and sigmaY respectively. 
# If only sigmaX is specified, sigmaY is taken as the same as sigmaX. 
# If both are given as zeros, they are calculated from the kernel size. 
# Gaussian blurring is highly effective in removing Gaussian noise from an image.

gb=cv2.GaussianBlur(img,ksize=(13,13),sigmaX=70)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(gb),plt.title('Gaussian Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


# MEDIAN BLURRING

# cv.medianBlur() takes the median of all the pixels under the kernel area and the central element is replaced with this median value. 
# This is highly effective against salt-and-pepper noise in an image. 
# Interestingly, in the above filters, the central element is a newly calculated value which may be a pixel value in the image or a new value. 
# But in median blurring, the central element is always replaced by some pixel value in the image. 
# It reduces the noise effectively. Its kernel size should be a positive odd integer.

median=cv2.medianBlur(img,ksize=15)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('Median Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

# CREATE GAUSSIAN NOISY IMAGE

def gaussianNoise(image):
    
    row,col,chanel=img.shape
    mean=0
    variance=0.05
    standard_dev=variance**0.5
    
    gauss=np.random.normal(mean,standard_dev,(row,col,chanel))
    gauss=gauss.reshape(row,col,chanel)
    noisy=image+gauss
    
    return noisy

img=cv2.imread("style3.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
noisy_image=gaussianNoise(img)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(noisy_image),plt.title('Noisy')
plt.xticks([]), plt.yticks([])
plt.show()

# Now let's fix these noisy image
gb2=cv2.GaussianBlur(img,ksize=(3,3),sigmaX=7)

plt.figure()
plt.imshow(gb2)
plt.axis("off")
plt.title("Gauss 2")
plt.show()


# CREATE SALT-PEPPER NOISY IMAGE 

def saltPepperNoise(image):
    
    row,col,chanel=img.shape
    s_and_p=0.5
    amount=0.004
    
    noisy=np.copy(image)
    
    # SALT : White
    num_salt=np.ceil(amount*image.size*s_and_p)
    coords=[np.random.randint(0,i-1,int(num_salt)) for i in image.shape]
    noisy[coords]=1
    
    # PEPPER : Black
    num_pepper=np.ceil(amount*image.size*(1-s_and_p))
    coords=[np.random.randint(0,i-1,int(num_pepper)) for i in image.shape]
    noisy[coords]=0
    
    return noisy

spImage=saltPepperNoise(img)
    
plt.figure()
plt.imshow(gb2)
plt.axis("off")
plt.title("Salt & Pepper")
plt.show()

# Fix these noise with Median Blur

median2=cv2.medianBlur(spImage.astype(np.float32),ksize=3)

plt.figure()
plt.imshow(median2)
plt.axis("off")
plt.title("Median 2")
plt.show()
