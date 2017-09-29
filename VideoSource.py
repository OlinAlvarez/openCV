import cv2
import numpy as np

cap = cv2.VideoCapture(0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
while True:
        ret, frame = cap.read()
       	cv2.imshow('frame',frame)
	    cv2.imshow('im2',im2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
