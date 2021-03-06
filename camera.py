#import numpy as np
import cv2

#default way to open camera
cap = cv2.VideoCapture(0)
t,fr = cap.read()
ctr = 0
while t == False:
    t,fr = cap.read()

print(ctr)
while True:
    _,frame = cap.read()
    '''
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print "Frames per second using cap.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
        print "Frames per second using cap.get(cv2.CAP_PROP_FPS) : {0}".format(fps)


    #sets HSV filter onto the video feed
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #fps = cap.get(cv2.CV_CAP_PROP_FPS)
    #shows both the filterless video and HSV video
    '''
    cv2.imshow('frame', frame)
    #cv2.imshow('hsv', hsv)
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
