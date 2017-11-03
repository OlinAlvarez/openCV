import cv2
import glob

 neg_imgs = []
 for img in glob.glob("neg_images/*.jpg"):
     n= cv2.imread(img)
     neg_imgs.append(n)

for i in range(0,len(neg_imgs)):
    neg_imgs[i] = cv2.im
    cv2.imwrite(str(i+1) + ".jpg",neg_imgs[i])



