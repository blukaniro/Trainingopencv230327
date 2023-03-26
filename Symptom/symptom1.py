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

# find leaf
    hsvLeaf_min = np.array([0/2,40,0]) # H：0-360 convert 0-180
    hsvLeaf_max = np.array([360/2,255,255])
    leafArea = cv.inRange(img_hsv, hsvLeaf_min, hsvLeaf_max)
    leafArea_count = cv.countNonZero(leafArea)

# find symptom
    hsvSymp_min = np.array([10/2,25,50]) # H：0-360 convert 0-180
    hsvSymp_max = np.array([56/2,255,230])
    sympArea = cv.inRange(img_hsv, hsvSymp_min, hsvSymp_max)

# analysis
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
dfResults.to_csv('Results.csv', index=False)