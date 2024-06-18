import os
import cv2 as cv
import numpy as np


img = cv.imread(os.path.join('..','data','kobe.jpg'))


img_edge = cv.Canny(img, 100,200)


kernel = np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]], dtype=np.uint8)


#kernel = np.ones((3,3), dtype=np.uint8)

img_edge_dilate = cv.dilate(img_edge, kernel, iterations=1)

img_edge_erode = cv.erode(img_edge_dilate, kernel, iterations=1)

#cv.imshow('img',img)
cv.imshow('img_edge_erode',img_edge_erode)
cv.imshow('img_edge', img_edge)
cv.imshow('img_dilate', img_edge_dilate)

cv.waitKey(0)
cv.destroyAllWindows()
