# -*- coding: utf-8 -*-

# ライブラリ読み込み
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# グレースケール
img = cv.imread("leaf.tif") #画像読み込み
img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite("gray.tif", img1)

# 二値化
img2 = img1.copy()
threshold = 150
ret, img2 = cv.threshold(img2, threshold, 255, cv.THRESH_BINARY)
cv.imwrite("mono.tif", img2)

# HSV変換
img3 = img.copy()
img_hsv = cv.cvtColor(img3, cv.COLOR_BGR2HSV)
cv.imwrite("leaf_hsv.tif", img3)

# Lab変換
img4 = img.copy()
img_lab = cv.cvtColor(img4, cv.COLOR_BGR2Lab)
cv.imwrite("leaf_lab.tif", img4)

# ガウジアンフィルタ
img5 = img.copy()
img5 = cv.GaussianBlur(img5, ksize=(3,3), sigmaX=1.3)
cv.imwrite("leaf_gaussian.tif", img5)

# メディアンフィルタ
img6 = img.copy()
img6 = cv.medianBlur(img6, ksize=3)
cv.imwrite("leaf_median.jpg", img6)

# matplotlibを用いてヒストグラムを表示
img = cv.imread("leaf.tif").astype(np.float64)
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("hist.png")

# 最近傍補間で画像拡大
img = cv.imread("leaf.tif") #画像読み込み
img7 = img.copy()
img7 = cv.resize(img7, (int(img7.shape[1]*1.5), int(img7.shape[0]*1.5)), interpolation=cv.INTER_NEAREST)
cv.imwrite("leaf_big.jpg", img7)