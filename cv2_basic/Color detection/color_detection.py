import cv2 as cv
from util import get_limits
from PIL import Image

cap = cv.VideoCapture(0)

color = [86, 164, 242]
while True:
    ret, frame = cap.read()

    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color)

    mask = cv.inRange(hsvFrame, lowerLimit, upperLimit)


    mask_s = Image.fromarray(mask)

    bounding_box = mask_s.getbbox()

    if bounding_box is not None:
        x1, y1, x2, y2 = bounding_box
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow('mask', mask)
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

