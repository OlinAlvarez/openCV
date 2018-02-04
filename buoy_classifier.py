import math
import matplotlib.pyplot as plt
import cv2
import dg
import glob
import numpy as np
import utils as ut
import multiscale_detect as md
from sklearn import svm
from random import shuffle

'''Default sizes for hog'''

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

''' getting the hog'''
dims = (96,144) #the width and height of the images being used.
hog = cv2.HOGDescriptor(dims,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,    histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

pos_imgs = []
neg_imgs = []

for img in glob.glob('pos_images/*.jpg'):
    n = cv2.imread(img)
    pos_imgs.append(n)

for img in glob.glob('neg_images/*.jpg'):
    n = cv2.imread(img)
    neg_imgs.append(n)

def getFeaturesWithLabel(imgData, hog, dims, label):
 data = []
 for img in imgData:
     img = cv2.resize(img, dims)
     #for images with transparency layer, reduce to 3 layers
     feat = hog.compute(img[:,:,:3])
     data.append((feat, label))
 return data

pdata = getFeaturesWithLabel(pos_imgs, hog, dims, 1)
ndata = getFeaturesWithLabel(neg_imgs, hog, dims, 0)

data = pdata + ndata
shuffle(data)

feat, labels =  map(list, zip(*data))
feat = [x.flatten() for x in feat]

sample_size = len(feat)
train_size = int(round(0.8*sample_size))

train_feat = np.array(feat[:train_size], np.float32)
test_feat = np.array(feat[train_size: sample_size], np.float32)
train_label = np.array(labels[:train_size])
test_label = np.array(labels[train_size:sample_size])
lsvm = svm.SVC(gamma=5, C=.12, kernel="linear", probability=True)
lsvm.fit(train_feat, train_label)

print lsvm.score(train_feat, train_label)
result = lsvm.predict(test_feat)
print "test accuracy ", lsvm.score(test_feat, test_label)
ut.printConfusionMatrix(result, test_label)

def getHogLSVM():
    return hog,lsvm
