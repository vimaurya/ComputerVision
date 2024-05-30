import os
import cv2 as cv

img = cv.imread(os.path.join('..','data','cat.jpg'))

img_cropped = img[50:183, 20:246]

print(img.shape)
print(img_cropped.shape)

cv.imshow('img', img)

cv.imshow('cropped_img', img_cropped)

cv.waitKey(0)