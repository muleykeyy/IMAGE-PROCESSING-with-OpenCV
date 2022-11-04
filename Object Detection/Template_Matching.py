import cv2
import matplotlib.pyplot as plt

img=cv2.imread("cat.jpg",0)
print(img.shape)

template=cv2.imread("cat_face.jpg",0)
print(template.shape)
h,w=template.shape

methods=["cv2.TM_CCOEFF","cv2.TM_CCOEFF_NORMED","cv2.TM_CCORR","cv2.TM_CCORR_NORMED","cv2.TM_SQDIFF","cv2.TM_SQDIFF_NORMED"]

for m in methods:
    method=eval(m) #eval() converts strings to function.
    
    pic=cv2.matchTemplate(img,template,method)
    print(pic.shape)
    
    min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(pic)
    
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left=min_loc
    else:
        top_left=max_loc
        
    bottom_right=(top_left[0]+w,top_left[1]+h)
    cv2.rectangle(img,top_left,bottom_right,255,2)
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(pic, cmap="gray")
    plt.title("Match")
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(img, cmap="gray")
    plt.title("Detected")
    plt.axis("off")   
    plt.suptitle(m)
    plt.show()