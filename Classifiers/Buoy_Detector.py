import numpy as np
import cv2

buoy_cascade = cv2.CascadeClassifier('./data/cascade.xml')
print(buoy_cascade)
img = cv2.imread('./y_pos/yellow_buoy1.jpg')
gray = cv2.imread('./y_pos/yellow_buoy1.jpg',0)

buoy = buoy_cascade.detectMultiScale(gray,1.3,5)

for(x,y,w,h) in buoy:
	print(x,y,w,h)
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
