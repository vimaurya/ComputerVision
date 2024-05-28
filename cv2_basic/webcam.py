import cv2 as cv

# read webcam
cam = cv.VideoCapture(0) #Value depends on number of cam

#visualize webcam

while True:

    ret, frame = cam.read()

    cv.imshow('frame', frame)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
