import cv2
import matplotlib.pyplot as plt
import numpy as np
# MORPHOLOGICAL OPERATIONS

# Morphological operations are one of the Image processing techniques that processes image based on shape. 
# This processing strategy is usually performed on binary images.

# 1) EROSION

# Erosion primarily involves eroding the outer surface (the foreground) of the image. 
# As binary images only contain two pixels 0 and 255, it primarily involves eroding the foreground of the image and it is suggested to have the foreground as white. 
# The thickness of erosion depends on the size and shape of the defined kernel.

img=cv2.imread("letter.jpg")
plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Original Image")

kernel=np.ones((5,5),dtype=np.uint8)
erosion=cv2.erode(img,kernel,iterations=2)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(erosion),plt.title('Eroded')
plt.xticks([]), plt.yticks([])
plt.show()

# 2) DILATION

# Dilation involves dilating the outer surface (the foreground) of the image. 
# As binary images only contain two pixels 0 and 255, it primarily involves expanding the foreground of the image and it is suggested to have the foreground as white. 
# The thickness of erosion depends on the size and shape of the defined kernel.


kernel=np.ones((5,5),dtype=np.uint8)
dilation=cv2.dilate(img,kernel,iterations=2)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dilation),plt.title('Dilated')
plt.xticks([]), plt.yticks([])
plt.show()

# 3) OPENING

# Opening involves erosion followed by dilation in the outer surface (the foreground) of the image. 
# All the above-said constraints for erosion and dilation applies here. It is a blend of the two prime methods. 
# It is generally used to remove the noise in the image.

# To see opening let's create white noise

whiteNoise=np.random.randint(0,2,size=img.shape[:3])
whiteNoise = whiteNoise * 255

plt.figure()
plt.imshow(whiteNoise, cmap="gray")
plt.axis("off")
plt.title("White Noise")

noise_img=whiteNoise + img

plt.figure()
plt.imshow(noise_img, cmap="gray")
plt.axis("off")
plt.title("White Noisy Image")

opening=cv2.morphologyEx(noise_img.astype(np.float32),cv2.MORPH_OPEN,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(opening),plt.title('Opening')
plt.xticks([]), plt.yticks([])
plt.show()

# CLOSING

# Closing involves dilation followed by erosion in the outer surface (the foreground) of the image. 
# All the above-said constraints for erosion and dilation applies here. 
# It is a blend of the two prime methods. It is generally used to remove the noise in the image.

# To see opening let's create black noise

blackeNoise=np.random.randint(0,2,size=img.shape[:3])
blackeNoise = blackeNoise * -255

plt.figure()
plt.imshow(blackeNoise, cmap="gray")
plt.axis("off")
plt.title("Black Noise")

noise_img2=blackeNoise + img

plt.figure()
plt.imshow(noise_img2, cmap="gray")
plt.axis("off")
plt.title("Black Noisy Image")

closing=cv2.morphologyEx(noise_img2.astype(np.float32),cv2.MORPH_CLOSE,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(closing),plt.title('Closing')
plt.xticks([]), plt.yticks([])
plt.show()

# Morphological Gradient

# Morphological gradient is slightly different than the other operations, because, 
# the morphological gradient first applies erosion and dilation individually on the image and then computes the difference between the eroded and dilated image. 
# The output will be an outline of the given image.

gradient= cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(gradient),plt.title('Gradient')
plt.xticks([]), plt.yticks([])
plt.show()
