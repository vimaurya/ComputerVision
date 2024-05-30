import os
import cv2 as cv

img = cv.imread(os.path.join('..', 'data', 'bear.jpg'))

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(img_gray, 80, 255, cv.THRESH_BINARY)

#thresh = cv.blur(thresh, (10,10))

#ret, thresh = cv.threshold(thresh, 80, 255, cv.THRESH_BINARY)


cv.imshow('img', img)
#cv.imshow('img_gray', img_gray)
cv.imshow('img_threshold', thresh)

cv.waitKey(0)
cv.destroyAllWindows()