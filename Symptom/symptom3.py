# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import cv2 as cv

os.chdir ("./Leaf/")
PhotoList=glob.glob('*.tif')
PhotoListLength=len(PhotoList)

Leaf = []
Symp = []
Percent = []
for i in range (PhotoListLength):
    os.chdir ("../Leaf/")
    img_original = cv.imread(PhotoList[i])
    img_hsv = cv.cvtColor(img_original, cv.COLOR_BGR2HSV)
    img_Lab = cv.cvtColor(img_original, cv.COLOR_BGR2Lab)

#find leaf
    hsvLeaf_min = np.array([0/2,40,0]) # H：0-360 convert 0-180
    hsvLeaf_max = np.array([360/2,255,255])
    leafArea = cv.inRange(img_hsv, hsvLeaf_min, hsvLeaf_max)

# find symptom
    img_Lab_L, img_Lab_a, img_Lab_b = cv.split(img_Lab)
    img_Lab_mod = np.where(img_Lab_b<1.1*img_Lab_a+51, 1, 0)\
        +np.where(img_Lab_b>1.2*img_Lab_L-80, 1, 0)
    sympArea = np.where(img_Lab_mod==2, 255, 0)
    sympArea =sympArea.astype(np.uint8)

# analysis
    leafArea_count = cv.countNonZero(leafArea)
    sympArea_count = cv.countNonZero(sympArea)
    sympAreaPercent = sympArea_count/leafArea_count*100
    Leaf.append(leafArea_count)
    Symp.append(sympArea_count)
    Percent.append(sympAreaPercent)

# visualization
    leafArea_inv = cv.bitwise_not(leafArea) # invert
    sympArea_inv = cv.bitwise_not(sympArea) # invert
    img_Leaf = cv.bitwise_and (img_original, img_original, mask = leafArea_inv) #AND
    img_Symp = cv.bitwise_and(img_original, img_original, mask = sympArea_inv) #AND

# export images
    os.chdir ("../LeafOutput/")
    cv.imwrite(str(PhotoList[i][:-4])+"_leaf.tif",img_Leaf)
    os.chdir ("../SympOutput/")
    cv.imwrite(str(PhotoList[i][:-4])+"_symp.tif",img_Symp)

# export data
os.chdir ("../")
dfResults = pd.DataFrame({'PhotoName': PhotoList,
                                            'Leaf': Leaf,
                                            'Symp': Symp,
                                            'Percent': Percent})
dfResults.to_csv('Results3.csv', index=False)