import sys
import cv2
import time
from PIL import Image
from hoosFace import hoosFace

classifier = hoosFace()
camera = cv2.VideoCapture(0)

while True:
    image = camera.read()
    img = Image.fromarray(image)
    name = classifier.name_face(img)
    cv2.putText(image, name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow('Camera output', image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
camera.release()