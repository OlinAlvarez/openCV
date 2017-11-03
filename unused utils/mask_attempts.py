import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('yellow_buoy_images/yellow_buoy1.jpg',1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

kernel = np.ones((5,5), np.float32) / 32
dst = cv2.filter2D(hsv,-1,kernel)

hist,bins =  np.histogram(dst.flatten(),256,[0,256])

low_bound = np.array([35,50,50])
hi_bound = np.array([50,100,100])

mask = cv2.inRange(img, low_bound, hi_bound)
mask_hsv = cv2.inRange(hsv, low_bound, hi_bound)
mask_dst = cv2.inRange(dst, low_bound, hi_bound)

while True:
    cv2.imshow('hsv',hsv)
    cv2.imshow('mask_hsv', mask_hsv)
    cv2.imshow('dst',dst)
    cv2.imshow('mask dst', mask_dst)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.imwrite('mask_pics/masked_buoypic.jpg',mask)
cv2.imwrite('mask_pics/hsv_mask.jpg',hsv) 
