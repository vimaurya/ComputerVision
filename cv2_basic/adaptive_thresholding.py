import os
import cv2 as cv

img = cv.imread(os.path.join('..','data','handwriting1.jpg'))

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img_gray, 120, 255, cv.THRESH_BINARY)


thresh_adaptive = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 20)
cv.imshow('img', img)
cv.imshow('img_thresh', thresh)
cv.imshow('img_adaptive_thresh', thresh_adaptive)
cv.waitKey(0)
cv.destroyAllWindows()