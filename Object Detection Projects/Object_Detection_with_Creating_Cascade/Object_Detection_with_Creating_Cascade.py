import cv2
import os

"""
- Create a Data Set n,p

- Cascade Trainer: https://amin-ahmadi.com/cascade-trainer-gui/

- With Cascade Tranier we can create our cascade

- Detection algorithm with cascade


"""
# CREATE IMAGE DATA FROM CAMERA
path="images"

imgWidth=180
imgHeight=120

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,180)

global countFolder

def saveDataFunc():
    global countFolder
    countFolder=0
    
    while os.path.exists(path+str(countFolder)):
        countFolder+=1
    os.makedirs(path+str(countFolder))
saveDataFunc()

count=0
countSave=0

while True:
    succes,img=cap.read()
    
    if succes:
        img=cv2.resize(img,(imgWidth,imgHeight))
        
        if count % 5 == 0:
            cv2.imwrite(path+str(countFolder)+"/"+str(countSave)+"_"+".png",img)
            countSave+=1
            print(countSave)
        count+=1
        cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):break 
cap.release()
cv2.destroyAllWindows()

#%% 
# After generating the cascade.xml file from Cascade Tranier, we can now start modeling

import cv2
objectName="Pen"
frameWidth=280
frameHeight=360
color=(255,0,255)

cap=cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)

def empty(a): pass
    

# Trackbar : For scale and neighbor

cv2.namedWindow("Result")
cv2.resizeWindow("Result",frameWidth,frameHeight+100)
cv2.createTrackbar("Scale","Result",400,1000,empty)
cv2.createTrackbar("Neighbor","Result",4,50,empty)

# CASCADE CLASSIFIER
cascade=cv2.CascadeClassifier("cascade.xml")

while True:
    succes,img=cap.read()
    
    if succes:
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        scaleVal=1+(cv2.getTrackbarPos("Scale","Result")/1000)
        neighbor=cv2.getTrackbarPos("Neighbor","Result")
        rects=cascade.detectMultiScale(gray,scaleVal,neighbor)
    
        for (x,y,w,h) in rects:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
            cv2.putText(img,objectName,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
        
        
        cv2.imshow("Result",img)
        
    if cv2.waitKey(1) & 0xFF == ord("q"):break

