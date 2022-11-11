import cv2
import os


files= os.listdir()
image_list=[]

for f in files:
    if f.endswith(".jpg"):
        image_list.append(f)
        
print(image_list)

# Histogram of Oriented Gradients Descriptor

hog= cv2.HOGDescriptor()

# Add SVM

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for imagePath in image_list:
    print(imagePath)
    
    image=cv2.imread(imagePath)
    
    (rects,weights)=hog.detectMultiScale(image,padding=(8,8),scale=1.05)
    
    for(x,y,w,h) in rects:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
     
      
    cv2.imshow("Pedestrian: ",image)
    
    if cv2.waitKey(0) & 0xFF == ord("q") : continue