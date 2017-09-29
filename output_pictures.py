import time
import numpy as np
import cv2

cap = cv2.VideoCapture(1)
interval  = 
while True:
    _,frame = cap.read()

    curr_time = int(round(time.time() * 1000))

    if curr_time % 15 == 0:
        cv2.imwrite('red_buoy_pics/red_bouy_' + str(curr_time) + '.jpg',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
