import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread("london.jpg",0)
plt.figure()
plt.imshow(img,cmap="gray")
plt.axis("off")
plt.show()


edges=cv2.Canny(image=img,threshold1=0,threshold2=255)
plt.figure()
plt.imshow(edges,cmap="gray")
plt.axis("off")
plt.show()

# To find best threshold values
median_value=np.median(img)
print(median_value)

low=int(max(0,(1-0.33)*median_value))
high=int(min(255,(1+0.33)*median_value))

print("LOW: ",low, "HIGH: ",high)

edges=cv2.Canny(image=img,threshold1=low,threshold2=high)
plt.figure()
plt.imshow(edges,cmap="gray")
plt.axis("off")
plt.show()

# Lets blur to lose edges in see

blured=cv2.blur(img,ksize=(3,3))
plt.figure()
plt.imshow(blured,cmap="gray")
plt.axis("off")
plt.title("Blured")
plt.show()

# Now try to find edges
median_value=np.median(blured)
print(median_value)

low=int(max(0,(1-0.33)*median_value))
high=int(min(255,(1+0.33)*median_value))

print("LOW Blured: ",low, "HIGH Blured: ",high)

edges=cv2.Canny(image=blured,threshold1=low,threshold2=high)
plt.figure()
plt.imshow(edges,cmap="gray")
plt.axis("off")
plt.title("Blured img Edges")
plt.show()