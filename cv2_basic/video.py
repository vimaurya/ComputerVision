import cv2 as cv
import os

video_path = os.path.join('..','data','cat.mp4')

video = cv.VideoCapture(video_path)

#Visualize
ret = True

while(ret):
    ret, frame = video.read()

    if ret:
        cv.imshow('frame', frame)
        cv.waitKey(10)


video.release()
cv.destroyAllWindows()