import os
import cv2 as cv

img = cv.imread(os.path.join('..','data','whiteboard.jpg'))
print(img.shape)

#Line
cv.line(img, (100,130), (100,340), (0,255,0), 3)

#Rectanle
cv.rectangle(img, (200, 130), (400, 340), (0,0,255), 3)

#Cirlce
cv.circle(img, (700,200),100, (255, 0, 0), 2)

#Text
cv.putText(img, "Don't look at this", (650, 500), cv.FONT_HERSHEY_SIMPLEX,0.5, (100, 100, 255), 2)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()