
from imutils.video import VideoStream
from datetime import datetime
from PIL import Image
import imutils
import time
import cv2
import os

from infer import FaceDetector

fd = FaceDetector()

print("Starting up camera ....")
vs = VideoStream(src=0).start()
time.sleep(2.0)
total = 0

print("Press p to name the customers in the webcam.")

while True:
    frame = vs.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=400)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("p"):
        img = Image.fromarray(frame)
        print(fd.name_face(img))

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()