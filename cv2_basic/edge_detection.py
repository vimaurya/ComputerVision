import os
import cv2 as cv
import numpy as np


img = cv.imread(os.path.join('..','data','kobe.jpg'))


img_edge = cv.Canny(img, 100,200)

img_edge_dilate = cv.dilate(img_edge, np.ones((3,3), dtype=np.int8))

img_edge_erode = cv.erode(img_edge_dilate, np.ones((3,3), dtype=np.int8))

cv.imshow('img',img)
cv.imshow('img_edge_erode',img_edge_erode)
cv.imshow('img_edge', img_edge)
cv.imshow('img_dilate', img_edge_dilate)

cv.waitKey(0)
cv.destroyAllWindows()
