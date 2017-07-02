import cv2, os
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.createLBPHFaceRecognizer()
cap = cv2.VideoCapture(0)
k = cv2.waitKey(30) & 0xff


def getImages (path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    labels = []
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        numberS = image_path.replace(path+"/", "")
        number = int(numberS.replace(".jpg", ""))
        faces = face_cascade.detectMultiScale(image)

        for (x, y, w, h) in faces:
            images.append(image)
            labels.append(number)
            cv2.imshow("Training...", image)
            cv2.waitKey(50)

    return images, labels


images, labels = getImages('./faces')


recognizer.train(images, np.array(labels))

while k != 27:

    ret, img = cap.read()
    img2 = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_gray = gray[y:y + h, x:x + w]
        predicted = 0
        predicted, conf = recognizer.predict(face_gray)
        print (predicted)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
cap.release()
cv2.destroyAllWindows()
