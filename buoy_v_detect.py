import utils
import math
import buoy_classifier as bc
import cv2
import numpy as np
cap = cv2.VideoCapture(1)
hog, lsvm = bc.getHogLSVM()
lower = [0,60,60]
upper = [60,255,255]
_,frame = cap.read()
height,width,lines = frame.shape
center = (width/2, height/2)
def preprocess(image, (lower,upper)):
    lower = np.array(lower, dtype='uint8')
    upper = np.array(upper, dtype='uint8')

    #apply smoothing
    kernel = np.ones((5,5), np.float32) / 32
    dst = cv2.filter2D( image, -1, kernel)

    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)

    return output, mask
def detect_buoy(im, boxes):
    true_boxes = []
    for box in boxes:
        x, y, w, h = box
        window = im[y:y+h, x:x+w, :]
        window = cv2.resize(window, bc.dims)
        feat = hog.compute(window)
        prob = lsvm.predict_proba(feat.reshape(1,-1))[0]
        if prob[1] > .1:
            true_boxes.append((box))
    return true_boxes

while True:
    _,frame = cap.read()

    pimage, mask = preprocess(frame, (lower,upper))
    imgray = cv2.cvtColor(pimage,cv2.COLOR_BGR2GRAY)
    flag,binaryImage = cv2.threshold(imgray, 85, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(binaryImage, 50, 150)

    im, contours, hieracrchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    copy = im.copy()
    #rgb = rgb = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes2 = [b for b in boxes if b[2]*b[3] > 400]
    for x, y, w, h in boxes2:
        cv2.rectangle(frame, (x,y), (x+w, y+h), utils.colors[utils.BLUE], 2)
    clone = im.copy()
    buoys = detect_buoy(frame,boxes2)
    for x, y, w, h in buoys:
        cv2.rectangle(clone, (x,y), (x+w,y+h), utils.colors[utils.BLUE], 2)
        cv2.line(frame,center,((x+(w/2)),(y+(h/2))),(255,0,0),3)
    #cv2.imshow('edges',edges)
    #cv2.imshow('pimage',pimage)
    cv2.imshow('clone',clone)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

