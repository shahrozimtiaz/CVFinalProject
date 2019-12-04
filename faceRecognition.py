import face_recognition as faceRecog

while(True):
    # knownFile = input("Enter the name of known file \n")
    unknownFileName = ''
    knownFileName = ''

    knownImage = faceRecog.load_image_file(knownFileName)
    unknownImage = faceRecog.load_image_file(unknownFileName)

    knownEncoding = faceRecog.face_encodings(knownImage)[0]
    unknownEncoding = faceRecog.face_encodings(unknownImage)[0]

    results = faceRecog.compare_faces([knownEncoding],unknownEncoding)

    print('The person is the same:', results[0])