import cv2
import numpy as np

low_bound = np.array([110,50,50])
hi_bound = np.array([130,255,255])

for i in range(1,126):
    img = cv2.imread('yellow_buoy_images/yellow_bouy1.jpg',1)
    mask_img = cv2.inRange(img,low_bound,hi_bound)

    while True:
        cv2.imshow('mask_img',mask_img)

        if cv.waitKey(0) & 0xFF == ord('q'):
            break

