import cv2

# IMAGE

# Importing Image:

img=cv2.imread("messi5.jpg",0) # 0 means: Grey Scale

# Show Image:
cv2.imshow("My Fav",img)

# Add Keyboard
k=cv2.waitKey(0) &0xFF  #Keyboard Key
if k==27: #27 means ESC
    cv2.destroyAllWindows()
elif k==ord("s"): #save pic
    cv2.imwrite("gray_pic.png",img)
    cv2.destroyAllWindows()
#%%    
# VIDEO
import cv2
import time

video_name="MOT17-04-DPM.mp4"

cap=cv2.VideoCapture(video_name)

# To see width and Height of video
print("Width:",cap.get(3))
print("Height: ",cap.get(4))

# To be sure about video is importing:
if cap.isOpened()==False:
    print("Video can NOT import")

# Read Video:
while True:
    ret,frame=cap.read()
    if ret==True:
        time.sleep(0.01) # If we cant use it, video going to fast
        cv2.imshow("Video",frame)
    else: break

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

#%%
# OPENING CAMERA & VIDEO RECORDING
import cv2

# OPENING CAMERA

cap=cv2.VideoCapture(0)

width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width,height)

# RECORDING VIDEO

writer=cv2.VideoWriter("my_record.mp4",cv2.VideoWriter_fourcc(*"DIVX"),20,(width,height))

while True:
    ret,frame=cap.read()
    cv2.imshow("Video",frame)
    
    #Save
    writer.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release()
writer.release()
cv2.destroyAllWindows()
