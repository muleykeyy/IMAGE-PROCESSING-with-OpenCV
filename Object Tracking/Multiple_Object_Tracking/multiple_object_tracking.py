import cv2

OpenCV_Object_Trackers={
    "boosting":cv2.legacy.TrackerBoosting_create,
    "csrt":cv2.TrackerCSRT_create,
    "kcf":cv2.TrackerKCF_create,
    "mosse":cv2.legacy.TrackerMOSSE_create,
    "mil":cv2.TrackerMIL_create,
    "tld":cv2.legacy.TrackerTLD_create,
    "medianFlow": cv2.legacy.TrackerMedianFlow_create
    }

tracker_name="medianFlow"
trackers=cv2.legacy.MultiTracker_create()

video_path="MOT17-04-DPM.mp4"
cap=cv2.VideoCapture(video_path)

fps=30
f=0

while True:
    
    ret,frame=cap.read()
    (H,W)=frame.shape[:2]
    frame=cv2.resize(frame,dsize=(960,540))
    
    (success,boxes)=trackers.update(frame)
    
    info=[("Tracker",tracker_name),("Success","Yes" if success else "No")]
    
    string_text=""
    
    for (i,(k,v)) in enumerate(info):
        text="{}: {}".format(k, v)
        string_text=string_text + text + " "
    cv2.putText(frame,string_text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(191,62,255),2)
    
    for box in boxes:
        (x,y,w,h)=[int(v) for v in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(191,62,255),2)
    
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) & 0xFF
    
    if key==ord("t"):
        box=cv2.selectROI("Frame",frame,fromCenter=False)
        
        tracker=OpenCV_Object_Trackers[tracker_name]()
        trackers.add(tracker,frame,box)
    
    elif key== ord("q"):break
    
    f=f+1

cap.release()
cv2.destroyAllWindows()
    
    



























