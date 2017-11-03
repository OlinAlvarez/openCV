import numpy as np
import cv2 
import numpy as np
#default way to open camera
cap = cv2.VideoCapture(0)
lower = np.array([0,86,0],dtype="uint8")
upper = np.array([90,255,90],dtype="uint8")


def preprocess(image): 
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    
    return output, mask

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    output,mask = preprocess(frame)
    #cv2.imshow('gray',gray)
    #cv2.imshow('binary',binary)
    #flag,binaryImage =  cv2.threshold(hsv,85,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('mask',mask)
    edges = cv2.Canny(mask, 50, 150)
    _, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    copy =  frame.copy()
    rgb = rgb = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
    boxes = [cv2.boundingRect(c) for c in contours]
	
    print len(boxes)
    boxes2 = [b for b in boxes if b[2]*b[3] >(75*75)]
    print len(boxes2)
	
    for x,y,w,h in boxes2:
        cv2.rectangle(frame, (x,y), (x+w,y+h) , (255,0,0), 2)
		
    cv2.imshow('frame',frame)
    #cv2.imshow('bin',binaryImage)
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
