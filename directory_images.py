from imutils.video import VideoStream
from datetime import datetime
import imutils
import time
import cv2
import os

name = input("Enter first name of person (ex. Guillermo): ")
name = name.lower().replace(" ","")
directory = "dataset/"+name

print("Starting up camera ....")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
vs = VideoStream(src=0).start()
time.sleep(2.0)
total = 0

print("Press p to the a picture")

while True:
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	rects = detector.detectMultiScale(
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))

	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("p"):
		if not os.path.exists(directory):
			os.makedirs(directory)
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		p = os.path.sep.join([directory, "{}.png".format(current_time)])
		cv2.imwrite(p, orig)
		total += 1
		print("Picture has been saved in directory")

	elif key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
