'''
 This file takes in 2 system arguements for filename and directory and
 saves images to that directory with that filename with time added at the end
 so that the images do not overwrite.

 it should be ran as follows:
     python savepics.py dirname filename
 videocapture 1 sets it to work with usb camera
 if it's set to 0 it will capture from webcam

 change secs to change the image capture rate
'''
import sys
import time
import cv2

cap = cv2.VideoCapture(1)
directory =  sys.argv[1]
name =  sys.argv[2]
secs = 5

while True:
    _,frame = cap.read()

    curr_time = int(round(time.time() * 1000))
    cv2.imshow('frame',frame)
    if curr_time % secs == 0:
        cv2.imwrite('yellow_buoy_images/yellow_buoy_' + str(curr_time) + '.jpg',frame)
        cv2.imwrite(directory +'/' + name + str(curr_time) + '.jpg',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
