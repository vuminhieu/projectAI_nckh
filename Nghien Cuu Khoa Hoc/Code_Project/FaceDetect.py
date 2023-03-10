import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
directory = r'E:\Learning\Hoc_Ki_Cuoi\Xu Ly Anh Python\Nghien Cuu Khoa Hoc\dataset'
face_id = input('\n Nhap ID Khuon Mat <return>  ==>   ')

print("\n [INFO] Khoi Tao Camera ...")
count = 0

while True:

    ret,img = cam.read()
    # img = cv2.flip(img,-1) # flip video image vertically
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img , (x,y) ,(x+w,y+h) , (255,0,0) , 2)
        count += 1
        cv2.imwrite("/Learning/Hoc_Ki_Cuoi/Xu Ly Anh Python/Nghien Cuu Khoa Hoc/dataset/User." + str(face_id) + '.' + str(count) + ".jpg" , img[y:y+h,x:x+w])

        cv2.imshow('image',img)

    if cv2.waitKey(100) & 0xff == ord('q'):
        break
    elif count >= 30:
        break

print("\n [INFO] Thoat")
cam.release()
cv2.destroyAllWindows()