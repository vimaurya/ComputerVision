import os
import cv2 as cv

img = cv.imread(os.path.join('..','data','cow.jpg'))

kernel_size = 5

#Averaging
img_blur = cv.blur(img, (kernel_size,kernel_size))

#Gaussian_blur
img_gauss_blur = cv.GaussianBlur(img, (kernel_size, kernel_size), 3)

#Median_blur
img_median_blur = cv.medianBlur(img, kernel_size)


cv.imshow('img_blur', img_blur)
cv.imshow('img', img)
cv.imshow('img_gauss_blur', img_gauss_blur)
cv.imshow('img_median_blur', img_median_blur)

cv.waitKey(0)
cv.destroyAllWindows()