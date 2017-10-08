import cv2
import numpy as np
from matploblib import pyplot as plt

img = cv2.imread('yellow_buoy_images/buoypic.jpg',1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

low_bound = np.array([35,50,50])
hi_bound = np.array([70,100,100])

mask = cv2.inRange(img, low_bound, hi_bound)
mask_hsv = cv2.inRange(hsv, low_bound, hi_bound)

while True:
    cv2.imshow('mask',mask)
    cv2.imshow('hsv',hsv)
    cv2.imshow('mask_hsv', mask_hsv)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.imwrite('masked_buoypic.jpg',mask)
cv2.imwrite('hsv_mask.jpg',hsv) 
