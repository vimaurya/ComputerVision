import cv2 as cv
import mediapipe as mp
import os

img_path = "C:\\Users\\Vikash maurya\\DataspellProjects\\computerVision\\data\\faces.jpg"
img = cv.imread(img_path)

if img is None:
    raise "Error: can not load the image"

H, W, _ = img.shape
print(W, H)


mp_face_detect = mp.solutions.face_detection

with mp_face_detect.FaceDetection(min_detection_confidence=0.3, model_selection=0) as face_detection:
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections:
        for detection in out.detections:
                location_data = detection.location_data
                bounding_box = location_data.relative_bounding_box

                x1, y1, w, h = bounding_box.xmin, bounding_box.ymin, bounding_box.width, bounding_box.height

                x1 = int(x1*W)
                y1 = int(y1*H)
                w = int(w*W)
                h = int(h*H)

                kernel_size = 80
                img[y1:y1+h, x1:x1+w, :] = cv.blur(img[y1:y1+h, x1:x1+w, :], (kernel_size, kernel_size))

        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()


output_path = "./result"

if not os.path.exists(output_path):
    os.makedirs(output_path)

cv.imwrite(os.path.join(output_path,'result.jpg'), img)