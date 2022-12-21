import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

"""
https://motchallenge.net/data/MOT17/
https://arxiv.org/pdf/1603.00831.pdf
"""


col_list=["frame_number","identity_number","left","top","width","height","score","class","visibility"]

data=pd.read_csv("gt.txt",names=col_list)

# To see classes in data set: 
plt.figure()
sns.countplot(data["class"])

####################################### CLASSES################################
# 1: Pedestrian
# 2: Person on Vehicle
# 3: Car
# 4: Bicycle 
# 5: Motorbike 
# 6: Non motorized vehicle 
# 7: Static person 
# 8: Distractor 
# 9: Occluder 
# 10: Occluder on the ground 
# 11: Occluder full 
# 12: Reflection


car=data[data["class"]==3]
video_path="MOT17-13-SDP.mp4"

cap=cv2.VideoCapture(video_path)

id1=29
number_of_image=np.max(data["frame_number"])
fps=25
bound_box_list=[]

for i in range(number_of_image-1):
    ret,frame=cap.read()
    
    if ret:
        frame=cv2.resize(frame,dsize=(960,540))
        filter_id1=np.logical_and(car["frame_number"]==i+1, car["identity_number"]==id1)
        
        if len(car[filter_id1]) !=0:
            x=int(car[filter_id1].left.values[0]/2)
            y=int(car[filter_id1].top.values[0]/2)
            w=int(car[filter_id1].width.values[0]/2)
            h=int(car[filter_id1].height.values[0]/2)
            
            # Draw Box:
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(178,58,238),2)
            cv2.circle(frame, (int(x+w/2),int(y+h/2)),2,(178,58,238),-1)
                
                # Add frame, x,y,width,height, center_x, center_y
            bound_box_list.append([i,x,y,w,h,int(x+w/2),int(y+h/2)])
                
        cv2.putText(frame, "Frame Number"+str(i+1),(10,30),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(191,62,255),2)
        cv2.imshow("frame",frame)
            
        if cv2.waitKey(1) & 0xFF == ord("q"):break
    else: break
    
cap.release()
cv2.destroyAllWindows()

gt2=pd.DataFrame(bound_box_list,columns=["frame_no","x","y","width","height","center_x","center_y"])
gt2.to_csv("my_gt.txt",index=False)




























