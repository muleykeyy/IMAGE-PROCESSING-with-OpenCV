import cv2
import numpy as np
from collections import deque

buffer_size=16
pts=deque(maxlen=buffer_size)


# BLUE COLOR RANGE HSV

blueLower=(90,50,70)
blueUpper=(128,255,255)

# Capture

cap=cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:
    succes,imgOriginal=cap.read()
    
    if succes:
        
        blured=cv2.GaussianBlur(imgOriginal,(11,11),0)
        
        # HSV
        hsv=cv2.cvtColor(blured,cv2.COLOR_BGR2HSV)
        #cv2.imshow("HSV Image",hsv)
        
        # Mask for Blue
        mask=cv2.inRange(hsv,blueLower,blueUpper)
        
        # Clean Noises
        mask=cv2.erode(mask,None,iterations=2)
        mask=cv2.dilate(mask,None,iterations=2)
        #cv2.imshow("Clean Image",mask)
        
        # Contour
        
        contours,h=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        center=None
        
        if len(contours)>0:
            c=max(contours,key=cv2.contourArea) #Biggest
            rect=cv2.minAreaRect(c)
            ((x,y),(width,height),rotation)=rect
            s="x: {}, y: {}, width: {}, height: {}, rotation: {} ".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            print(s)
            
            # BOX
            box=cv2.boxPoints(rect)
            box=np.int64(box)
            
            # MOMENT
            M=cv2.moments(c)
            center=(int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
            
            # PLOT
            cv2.drawContours(imgOriginal,[box],0,(0,255,255),2)
            cv2.circle(imgOriginal,center,5,(255,0,255),-1)
            cv2.putText(imgOriginal,s,(50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),2)
            
        # Deque: Where is object and where is it going?
        pts.appendleft(center)
        
        for i in range(1,len(pts)):
            if pts[i-1] is None or pts[i] is None: continue
            cv2.line(imgOriginal,pts[i-1],pts[i],(0,255,0),3)
            
            
            
        cv2.imshow("DETECT",imgOriginal)
            
    if cv2.waitKey(1) & 0xFF==ord("q"):break
