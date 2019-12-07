# import classify
import cv2
import preprocess

# classifier = classify.Classify()
preprocessor = preprocess.PreProcessor()

camera = cv2.VideoCapture(0)
while True:
    return_value, image = camera.read()
    bb = preprocessor.align(image)
    scaled_img = 'temp.png'
    cv2.rectangle(image, (bb[0],bb[1]), (bb[2],bb[3]), (0, 255, 0), 5)
    # name = classifier.predict(scaled_img)
    name = 'label'
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, name, (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()