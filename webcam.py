from imutils.video import VideoStream
from PIL import Image
import imutils
import time
import cv2
import preprocess
import os

from hoosFace import HoosFace

print('Initializing ....')
preprocessor = preprocess.PreProcessor()
classifier = HoosFace()

print("Starting up camera ....")
vs = VideoStream(src=0).start()
time.sleep(2.0)
total = 0

print("Press q to quit.")

while True:
    frame = vs.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=400)

    bb = preprocessor.align(frame)
    scaled_img = 'temp.png'
    cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 5)
    name = classifier.name_face(Image.fromarray(frame))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, name, (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
os.remove(scaled_img)
