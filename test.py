import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = 'data/kobe.jpg'
color_image = cv2.imread(image_path)

hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

h, s, v = cv2.split(hsv_image)
v_dilated = cv2.dilate(v, kernel, iterations=1)
v_eroded = cv2.erode(v, kernel, iterations=1)

hsv_dilated = cv2.merge([h, s, v_dilated])
hsv_eroded = cv2.merge([h, s, v_eroded])

dilated_color_image = cv2.cvtColor(hsv_dilated, cv2.COLOR_HSV2BGR)
eroded_color_image = cv2.cvtColor(hsv_eroded, cv2.COLOR_HSV2BGR)

titles = ['Original Color Image', 'Dilated Color Image (V Channel)', 'Eroded Color Image (V Channel)']
images = [color_image, dilated_color_image, eroded_color_image]

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
