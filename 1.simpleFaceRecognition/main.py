import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.createLBPHFaceRecognizer()
capture = cv2.VideoCapture(0)
k = 0

while k != 27:
    k = cv2.waitKey(30) & 0xff
    succes,image = capture.read()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray_image, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('img', image)


capture.release()
cv2.destroyAllWindows()
