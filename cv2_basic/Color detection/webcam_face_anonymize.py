import cv2 as cv
import mediapipe as mp

cam = cv.VideoCapture(0)

while cam.isOpened():

    ret, frame = cam.read()

    H, W, _ = frame.shape

    mp_face_detect = mp.solutions.face_detection

    with mp_face_detect.FaceDetection(min_detection_confidence=0.3, model_selection=0) as face_detection:
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
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
                frame[y1:y1+h, x1:x1+w, :] = cv.blur(frame[y1:y1+h, x1:x1+w, :], (kernel_size, kernel_size))

            cv.imshow('webcam', frame)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break



cam.release()
cv.destroyAllWindows()
