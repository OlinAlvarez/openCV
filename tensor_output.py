import numpy as np
import cv2

#default way to open camera
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()	
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    fps = cap.get(cv2.CV_CAP_PROP_FPS) 
    #shows both the filterless video and HSV video
    cv2.imshow('frame', frame)
    cv2.imshow('hsv', hsv)
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
