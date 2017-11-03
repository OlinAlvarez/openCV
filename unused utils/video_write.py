import numpy as np
import cv2

cam_FPS = 20

cap = cv2.VideoCapture(1)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

#cap.set(cv2.cv.CV_CAP_PROP_FPS, cam_FPS)

while True:
    _, frame = cap.read()
#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out.write(frame)
    cv2.imshow('frame', frame)
#    cv2.imshow('hsv', hsv)
        
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
