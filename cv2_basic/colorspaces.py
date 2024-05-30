import os
import cv2 as cv

img = cv.imread(os.path.join('..','data','bird.jpg'))


img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

print(img.shape)
print(img_rgb.shape)
print(img_gray.shape)

cv.imshow('img', img)
cv.imshow('img_rgb', img_rgb)
cv.imshow('img_gray', img_gray)
cv.waitKey(0)