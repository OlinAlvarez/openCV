import numpy as np
import cv2

cap = cv2.VideoCapture(1)
lower_bound = [0,0,0]
upper_bound = [100,100,255]
lower = np.array(lower_bound, dtype='uint8')
upper = np.array(upper_bound, dtype='uint8')

while True:
    _,frame = cap.read()
    mask = cv2.inRange(frame,lower,upper)
    bitwise_img = cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('bitwise_img',bitwise_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
