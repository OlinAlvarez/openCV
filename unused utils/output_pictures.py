import time
import numpy as np
import cv2

cap = cv2.VideoCapture(1)

while True:
    _,frame = cap.read()

    curr_time = int(round(time.time() * 1000))
    cv2.imshow('frame',frame)
    if curr_time % 5 == 0:
        cv2.imwrite('water_pics/buoy_' + str(curr_time) + '.jpg',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
