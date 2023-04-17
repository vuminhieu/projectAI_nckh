import cv2
import numpy as np
import os
import time , datetime
import mysql.connector
import yagmail

#connect to email server
# yag = yagmail.SMTP("quoctoanahihi.123@gmail.com","bxsdcalcnqryqjhu")

recoginzer = cv2.face.LBPHFaceRecognizer_create()
recoginzer.read("E:/Learning/Hoc_Ki_Cuoi/Xu Ly Anh Python/Nghien Cuu Khoa Hoc/trainer/trainer.xml")
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

file_excel_write = open("E:/Learning/Hoc_Ki_Cuoi/Xu Ly Anh Python/Nghien Cuu Khoa Hoc/Excel/face_recognition.csv" , mode="w", encoding="utf-8-sig")
file_excel_read = open("E:/Learning/Hoc_Ki_Cuoi/Xu Ly Anh Python/Nghien Cuu Khoa Hoc/Excel/timeTable_sv.csv" , mode="r", encoding="utf-8-sig")

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
pTime = 0

justscanned = False
cnt = 0
pause_cnt = 0

mydb = mysql.connector.connect(
        host='localhost',
        user='root',
        password='08112001',
        port='3306',
        database='face_recognition'
    )
cursor = mydb.cursor()

cam = cv2.VideoCapture(0)

#doc file tư Excel xuong
header = file_excel_read.readline()
file_excel_write.write(header)

def getProfile(id) :
    query = ("SELECT * FROM table_sv WHERE msv=" + str(id))
    cursor.execute(query)
    users = cursor.fetchall()
    profile = None
    for user in users :
        profile = user
    return profile

while True :
    ret , img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray)

    # get current time :
    now = datetime.datetime.now()
    date = now.strftime("%d-%m-%Y")
    time_now = now.strftime("%H:%M:%S")

    # show fps :
    cTime= time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}' , (10,40) , font , 1 , (0,255,0) , 2)

    pause_cnt += 1

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, confidence = recoginzer.predict(gray[y:y + h, x:x + w])
        # predictions = model.predict(gray[y:y + h, x:x + w])

        confidence = int(100 * (1 - confidence / 300))

        # nhận diện với predict:
        if confidence > 70:
            user = getProfile(id)

            cnt += 1
            n = (100 / 30) * cnt
            w_filled = (cnt / 30) * w

            cv2.putText(img, str(user[1]), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(user[0]), (x + 5, y - 40), font, 1, (255, 255, 255), 2)

            if int(cnt) == 30:
                cnt = 0
                # save Excel :
                row_student = str(user[0]) + ',' + user[1] + ',' + str(user[2]) + ',' + user[3] + ',' + user[4] + ',' + \
                          user[5] + ',' + user[6] + ',' + str(date) + ',' + str(time_now) + '\n'

                row_cr = row_student.split(",")[0]
                # row_new = row_student
                file_excel_write.writelines(row_student)

                # send email :
                to = user[6]
                subject = "Xác Nhận Điểm Danh Lớp THCN2"
                body = "Xin Chào "+ user[1] +"-"+ str(user[0]) +"\n" \
                        +user[1] + " đã điểm danh thành công vào lúc " +str(time_now)+ " , "+ str(date)
                print(body)
                # yagmail.send(to=to,subject=subject,contents=body)

                # justscanned = True
                pause_cnt = 0

        else:
            # if not justscanned:
            cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            # else:
            #     cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            if pause_cnt > 80:
                justscanned = False

    cv2.imshow('nhan dien khuon mat' , img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

print("\n [INFO] Thoat")
cam.release()
cv2.destroyAllWindows()
