import cv2
import os
import time
import mysql.connector
import numpy as np
from PIL import Image

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
directory = r'E:\Learning\Hoc_Ki_Cuoi\Xu Ly Anh Python\Nghien Cuu Khoa Hoc\dataset'

def InsertOrUpdate(msv,name,age,lop,grade,gender,email):

    mydb = mysql.connector.connect(
        host= 'localhost' ,
        user= 'root' ,
        password='08112001' ,
        port= '3306' ,
        database= 'face_recognition'
    )
    cursor = mydb.cursor()

    query = "INSERT INTO `table_sv` (`msv`, `name`, `age`, `class`, `grade`, `gender`,`email`) VALUES (" + str(msv) + ",'" + str(name) + "','" + str(age) + "','" + str(lop) + "','" + str(grade) + "','" + str(gender) + "','" + str(email) + "')"

    # query = "UPDATE `table_sv` SET `name` = '" + str(name) + "' , `age` = '" + str(age) + "' , `class` = '" + str(lop) + "' , `grade` = '" + str(grade) + "' , `gender` = '" + str(gender) + "' WHERE `msv` = '" + str(msv) + "'"

    cursor.execute(query)
    mydb.commit()

msv = input('\n Enter Your MSV <return>  ==>   ')
name = input('\n Enter Name  ==>   ')
age = input('\n Enter Age  ==>   ')
lop = input('\n Enter Class ==>   ')
grade = input('\n Enter Grade ==>   ')
gender = input('\n Enter Gender ==>   ')
email = input('\n Enter E-Mail ==>   ')

print("\n [INFO] Khoi Tao Camera ...")
count = 0
InsertOrUpdate(msv,name,age,lop,grade,gender,email)

while True:

    ret,img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img , (x,y) ,(x+w,y+h) , (255,0,0) , 2)
        count += 1
        time.sleep(0.5)
        cv2.imwrite("/Learning/Hoc_Ki_Cuoi/Xu Ly Anh Python/Nghien Cuu Khoa Hoc/dataset/{}.".format(name) + str(msv) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
        cv2.imshow('image', img)

    if cv2.waitKey(100) & 0xff == ord('q'):
        break
    elif count >= 10:
        break

print("\n [INFO] Thoat")
cam.release()
cv2.destroyAllWindows()