import os
import cv2 as cv

#read image

image_path = os.path.join('..', 'data', 'cat.jpg')

img = cv.imread(image_path)


#Write image

cv.imwrite(os.path.join('..', 'data', 'catout.jpg'), img)


cv.imshow('bird', img)

cv.waitKey(5000)

