import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

img = cv2.imread('dice.png',-1)
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(grey,127,255,0)
plt.imshow(img)
#plt.imshow(thresh)
plt.show()
im2, contours, hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
plt.imshow(im2)
plt.show()
boxes = [ cv2.boundingRect(c) for c in contours]
dice = [b for b in boxes if b[2] * b[3] > 9000 and b[2]*b[3] < 200000]
dice = dice[::2]
print len(dice)
#create a dictionary and then add the numerical value of the die to the corresponding np array index.
count = 1;
dice_dict = {}
for i  in range(0,len(dice)):
    dice_dict[dice[i]] = -1
font  = cv2.FONT_HERSHEY_SIMPLEX
def center(die):
    return ( (die[0] + die[2] / 2) , (die[1] + die[3] / 2))
def getDots(window):
    bw_window = cv2.cvtColor(window,cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(bw_window,127,255,0)
    temp_im,conts,_ = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    dots = [ cv2.boundingRect(c) for c in conts]
    dots = [ d for d in dots if d[2] * d[3] < 12000]
    for x1,y1,w1,h1 in dots:
        cv2.rectangle(window,(x1,y1),(x1+w1,y1+h1),(0,255,255),3)
    #plt.imshow(window)
    #print "number of dots ",len(dots)
    #plt.show()
    return len(dots)

for x, y, w, h in dice:
    window = img[y+30:y+h-30, x+30:x+w-30, :]
    cv2.rectangle(img, (x,y),(x+w, y+h), (0, 255, 0), 5)
    dice_dict[(x,y,w,h)] = getDots(window)
    cv2.putText(img,str(dice_dict[(x,y,w,h)]),(x , y),font,2,(255,0,0),5,cv2.LINE_AA)

print dice_dict
path_centers = [] #holds the center points for where the points I want to draw`

for i in range(0,6):
    for j in range (i+1,6):
        if(dice_dict[dice[i]] + dice_dict[dice[j]] == 7 or dice_dict[dice[i]] + dice_dict[dice[j]] == 11):
            path_centers.append(( center(dice[i]) , center(dice[j])))
            #path_centers.append( ( (dice[i][0] + dice[i][2] / 2, dice[i][1] + dice[i][3] / 2)  , (dice[j][0] + dice[j][2] / 2, dice[j][1] + dice[j][3] / 2)))
print len(path_centers),path_centers

for i in range(0,len(path_centers)):
    cv2.line(img,path_centers[i][0],path_centers[i][1],(0,0,255),2)




plt.imshow(img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
