import cv2
from tracker import EuclideanDistTracker
video= cv2.VideoCapture('highway.mp4')

object=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)
tracker=EuclideanDistTracker()
while True:
    ret, frame=video.read()
    roi = frame[340:720, 500:800]
 # 1. Object Detcetion
    mask=object.apply(roi)
    _,mask=cv2.threshold(mask,254,255,cv2.THRESH_BINARY
                         )
    cont,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    h,w,_=frame.shape

    detector=[]
    for conts in cont:
      area=cv2.contourArea(conts)
      if area>100:
        # cv2.drawContours(frame,conts,-1,(0,255,0),2)
          x,y,w,h=cv2.boundingRect(conts)

          detector.append([x,y,h,w])

    b_ids = tracker.update(detector)
    for ids in b_ids:

        x,y,h,w,id=ids
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(roi,str(id),(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
    cv2.imshow("Frame", frame)
    # cv2.imshow("Mask",mask)
    # cv2.imshow("Frame", roi)
    key=cv2.waitKey(30)
    if key==27:
        break

video.release()
cv2.destroyWindow()