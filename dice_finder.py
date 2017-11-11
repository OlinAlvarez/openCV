import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

img = cv2.imread('dice.png',-1)
cv2.line(img,(0,0),(511,511),(5,233,0),5)
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(grey,127,255,0)
im2, contours, hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
plt.show(contours)
plt.show()
boxes = [ cv2.boundingRect(c) for c in contours]
dice = [b for b in boxes if b[2] * b[3] > 9000 and b[2]*b[3] < 200000]
print len(dice)
#create a dictionary and then add the numerical value of the die to the corresponding np array index.
count = 1;
dice_dict = {}
for i  in range(0,len(dice)):
    dice_dict[dice[i]] = -1

print dice_dict
font  = cv2.FONT_HERSHEY_SIMPLEX
for x, y, w, h in dice:
    print (x,y)
    window = img[y:y+h, x:x+2, :]
    cv2.putText(img,str(count),(x + (w/2), y + (h/2)),80,font,(255,0,0),5,cv2.LINE_AA)
    cv2.rectangle(img, (x,y),(x+w, y+h), (0, 255, 0), 5)
    count = count + 1
#def getDots(window):
cv2.line(img,(0,0),(511,511),(255,0,0),5)
plt.imshow(img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
