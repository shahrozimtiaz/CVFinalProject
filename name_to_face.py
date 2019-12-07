from imutils.video import VideoStream
from datetime import datetime
import imutils
import time
import cv2
import os
from PIL import Image

print("Starting up camera ....")
vs = VideoStream(src=0).start()
time.sleep(2.0)
total = 0

print("Press p to the a picture")

while True:
    frame = vs.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=400)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord("p"):
        print(type(frame))
        img = Image.fromarray(frame)

        print(frame.shape)
        print("WOW!")

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()