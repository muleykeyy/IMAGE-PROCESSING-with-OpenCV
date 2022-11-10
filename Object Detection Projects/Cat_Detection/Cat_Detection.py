import cv2
import os

files=os.listdir()
print(files)


# Import Multiple Image
img_list=[]

for f in files:
    if f.startswith("cat_"):  # f.endwith(".jpg")
        img_list.append(f)
print("Cat Images: ",img_list)

# Visualize Images
for j in img_list:
    print(j)
    images=cv2.imread(j)
    gray=cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
    
    detector=cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    rects=detector.detectMultiScale(gray,scaleFactor=1.037,minNeighbors=3) #scaleFactor : Zoom
    
    for (i,(x,y,w,h))in enumerate(rects):
        cv2.rectangle(images,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.putText(images,"Cat {}".format(i+1),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)

    
    cv2.imshow(j,images)
    if cv2.waitKey(0) & 0xFF == ord("f"):continue