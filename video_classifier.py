import matplotlib.pyplot as plt
import dg #changed from Mark's code to just dg short for data generator
import cv2
import numpy as np
import utils as ut
import glob
import multiscale_detect as md
from sklearn import svm
from random import shuffle
#opening getting the positive and negative images.
pos_imgs = []
for img in glob.glob("pos_images/*.jpg"):
    n= cv2.imread(img)
    pos_imgs.append(n)

print len(pos_imgs)

neg_imgs = []
for img in glob.glob("neg_images/*.jpg"):
    n= cv2.imread(img)
    neg_imgs.append(n)

print len(neg_imgs)
def getFeaturesWithLabel(imgData, hog, dims, label):
    data = []
    for img in imgData:
        img = cv2.resize(img, dims)

        #for images with transparency layer, reduce to 3 layers
        feat = hog.compute(img[:,:,:3])

        data.append((feat, label))
    return data

minDim = 80
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = -1
histogramNormType = 0
L2HysThreshold = 2.1e-1
gammaCorrection = 0
nlevels = 64

dims = (96,144) #the width and height of the images being used.

hog = cv2.HOGDescriptor(dims,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                         histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

pdata = getFeaturesWithLabel(pos_imgs, hog, dims, 1)
ndata = getFeaturesWithLabel(neg_imgs, hog, dims, 0)

data = pdata + ndata
shuffle(data)

feat, labels = map(list, zip(*data))
feat = [x.flatten() for x in feat]

sample_size = len(feat)
train_size = int(round(0.8*sample_size))

train_feat = np.array(feat[:train_size], np.float32)
test_feat = np.array(feat[train_size: sample_size], np.float32)
train_label = np.array(labels[:train_size])
test_label = np.array(labels[train_size:sample_size])
lsvm = svm.SVC(gamma=5, C=.1, kernel="linear", probability=True)
lsvm.fit(train_feat, train_label)

print lsvm.score(train_feat, train_label)
result = lsvm.predict(test_feat)
print "test accuracy ", lsvm.score(test_feat, test_label)
ut.printConfusionMatrix(result, test_label)

def getHardNegativeFeatures(lsvm, hog, imgDir, step=1, scale=1.1):
    #get dataset of negative images to scan through
    negFiles = dg.getAllFiles(imgDir)
    negImgs = []
    falsePos_feat = []
    signs = 0
    nosigns = 0

    for f in negFiles:
        negImgs.append([f, cv2.imread(imgDir + "/" + f)])
    if len(negImgs) == 0:
        raise "No images found"

    #multiscale detect
    for row in negImgs:
        scales = md.pyramid(row[1], scale, minSize=(30, 30))
        winw = hog.winSize[1]
        winh = hog.winSize[0]
        for img in scales:
            #results = []

            for (x, y, window) in md.sliding_window(img, step, (winw, winh)):
                if window.shape[0] != winh or window.shape[1] != winw:
                    continue
                window = cv2.resize(window, (winh, winw))
                feat = hog.compute(window)
                result = lsvm.predict(feat.reshape(1,-1))
                if result == 1:
                    falsePos_feat.append(feat)
                    signs +=1
                else:
                    nosigns+=1

    falsePosFeat = [x.flatten() for x in falsePos_feat]
    return falsePosFeat

def preprocess(image, (lower, upper)):

    #boundary in bgr color scheme for opencv

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    #apply smoothing
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(image,-1,kernel)

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)

    output = cv2.bitwise_and(image, image, mask = mask)

    return output, mask
'''
r = np.matrix(im[:,:,2]).sum()
b = np.matrix(im[:,:,0]).sum()
g = np.matrix(im[:,:,1]).sum()
print r, g, b
red_ratio = float(r)/(b+g)
print red_ratio
red_val = int(red_ratio * 120)
'''
cap = cv2.VideoCapture(1)

while True:
    _,frame = cap.read()
    imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    flag, binaryImage = cv2.threshold(imgray, 85, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
    edges = cv2.Canny(binaryImage, 50, 150)
    im2, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    copy = frame.copy()
    rgb = rgb = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes2 = [b for b in boxes if b[2]*b[3] > 200]

    for x, y, w, h in boxes2:
        cv2.rectangle(rgb, (x,y),(x+w, y+h), (0, 255, 0), 2)
    plt.imshow(rgb)
    plt.show()

    real_signs = []

    for x, y, w, h in boxes2:
        #get slice at box:
        window = frame[y:y+h, x:x+w, :]
        window = cv2.resize(window, dims)
        feat = hog.compute(window)
        prob = lsvm.predict_proba(feat.reshape(1,-1))[0]
    #     plt.imshow(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))
    #     plt.show()
    #     print prob[1]
        if prob[1] > .1:
            real_signs.append((x,y,w,h))

    print len(real_signs)
    clone = frame.copy()

    colors = [(0,255,0),(0,0,0),(255,0,255),(255,0,0),(255,165,0),(255,255,255), (1, 1, 1)]
    for x, y, w, h in real_signs:
        cv2.rectangle(clone, (x, y), (x+w, y+h), colors[3], 2)

    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.show()
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

