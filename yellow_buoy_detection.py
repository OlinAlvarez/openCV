import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([55,147,118], dtype = np.uint8)
    upper_bound = np.array([173,255,255], dtype = np.uint8)
    
    mask = cv2.inRange(frame, lower_bound, upper_bound)

    cv2.imshow('mask',mask)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
