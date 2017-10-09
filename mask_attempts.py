import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('yellow_buoy_images/buoypic.jpg',1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

kernel = np.ones((5,5), np.float32) / 32
dst = cv2.filter2D(img,-1,kernel)

hist,bins =  np.histogram(img.flatten(),256,[0,256])

low_bound = np.array([35,50,50])
hi_bound = np.array([50,100,100])

mask = cv2.inRange(img, low_bound, hi_bound)
mask_hsv = cv2.inRange(hsv, low_bound, hi_bound)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]),plt.yticks([])
plt.show()
while True:
    cv2.imshow('mask',mask)
    cv2.imshow('hsv',hsv)
    cv2.imshow('mask_hsv', mask_hsv)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.imwrite('masked_buoypic.jpg',mask)
cv2.imwrite('hsv_mask.jpg',hsv) 
