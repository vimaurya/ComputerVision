import cv2 as cv
import numpy as np

img_path = 'data/faces.jpg'

img = cv.imread(img_path)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

kernel1 = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]], np.float32)

kernel2 = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]], np.float32)

horizontal = cv.filter2D(img_gray, 0, kernel1)

vertical = cv.filter2D(img_gray, 0, kernel2)

cv.imshow('original', img_gray)
cv.imshow('filter', horizontal)
cv.imshow('filter_vertical', vertical)
cv.waitKey(0)
cv.destroyAllWindows()
