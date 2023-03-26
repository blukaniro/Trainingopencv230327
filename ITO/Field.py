# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import cv2 as cv

import random # 必要ない

# 切り出し
os.chdir ("./Field/")
PhotoList=glob.glob("*.JPG")
for i in range (len(PhotoList)):
    img = cv.imread(PhotoList[i])
    height, width, channel = img.shape
    os.chdir ("../Point/")
    for j in range (50):
        x=random.uniform(0, height-10)
        y=random.uniform(0, width-10)
        X1=int(x)
        Y1=int(y)
        X2=int(X1+10)
        Y2=int(Y1+10)
        img2 = img[X1:X2,Y1:Y2]
        cv.imwrite(str(PhotoList[i][:-4])+str(j+1)+".JPG", img2)
    os.chdir ("../Field/")

# 解析
os.chdir ("../Point/")
PointList=glob.glob("*.JPG")
PointListLength=len(PointList)

Lab_b = []

for i in range (PointListLength):
    os.chdir ("../Point/")
    img_original = cv.imread(PointList[i])
    img_Lab = cv.cvtColor(img_original, cv.COLOR_BGR2Lab)

# find plants
    img_Lab_L, img_Lab_a, img_Lab_b = cv.split(img_Lab)
    L_true, a_true, b_true = 50, 0, 50
    L_mod, a_mod, b_mod = L_true/100*255, a_true+127, b_true+127 # need modify
    NoplantsArea = np.where(img_Lab_a<a_mod, img_Lab_a, 0)

# 図示
    img_plants = cv.bitwise_and(img_original, img_original, mask = NoplantsArea)
    os.chdir ("../Output/")
    cv.imwrite(str(PointList[i][:-4])+".JPG",img_plants)

# 解析
    avg_b=0
    for j in range(img_Lab_b.shape[0]):
        for k in range(img_Lab_b.shape[1]):
            if img_Lab_a[j,k]<a_mod:
                avg_b += img_Lab_b[j,k]
    Lab_b.append(avg_b/img_Lab_b.size)

# 書き出し
os.chdir ("../")
dfResults = pd.DataFrame({'PointName': PointList,
                                            'b_on_a0': Lab_b})
dfResults.to_csv('Results.csv', index=False)
