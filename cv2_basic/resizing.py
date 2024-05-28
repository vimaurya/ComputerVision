import os
import cv2 as cv

img = cv.imread(os.path.join('..','data','cat.jpg'))


img_resized = cv.resize(img, (552, 366))

print(img.shape)
print(img_resized.shape)

cv.imshow('img', img)
cv.imshow('img_resized', img_resized)

cv.waitKey(0)