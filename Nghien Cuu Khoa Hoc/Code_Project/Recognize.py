import cv2
import numpy as np
import os

recoginzer = cv2.face.LBPHFaceRecognizer_create()
recoginzer.read("E:/Learning/Hoc_Ki_Cuoi/Xu Ly Anh Python/Nghien Cuu Khoa Hoc/trainer/trainer.yml")
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['Quoc Toan' , '2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8' ,'9']

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True :
    ret , img = cam.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id,confidence = recoginzer.predict(gray[y:y+h , x:x+w])

        if (confidence < 100) :
            id = names[id]
            confidence =" {0}%".format(round(100 - confidence))
        else:
            id = 'unknown'
            confidence =" {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id) ,(x+5 , y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    cv2.imshow('nhan dien khuon mat' , img)

    if cv2.waitKey(10) & 0xff == ord('q'):
        break

print("\n [INFO] Thoat")
cam.release()
cv2.destroyAllWindows()