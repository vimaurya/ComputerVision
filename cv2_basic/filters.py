import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../data/bear.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

laplacian = cv2.Laplacian(image, cv2.CV_64F)

median = cv2.medianBlur(image, 5)

plt.figure(figsize=(12, 8))

plt.subplot(231), plt.imshow(image), plt.title('Original')
plt.subplot(232), plt.imshow(gaussian_blur), plt.title('Gaussian Blur')
plt.subplot(233), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel Edge Detection')
plt.subplot(234), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian Edge Detection')
plt.subplot(235), plt.imshow(median), plt.title('Median Filter')

plt.show()
