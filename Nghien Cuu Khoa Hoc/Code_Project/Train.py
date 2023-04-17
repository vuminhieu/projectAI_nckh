import cv2
import numpy as np
from PIL import Image
import os

directory = r'E:\Learning\Hoc_Ki_Cuoi\Xu Ly Anh Python\Nghien Cuu Khoa Hoc\dataset\dataImg'
recoginzer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getImagesandLabels(path) :
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths :
        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces = face_detector.detectMultiScale(img_numpy)

        for(x,y,w,h) in faces :
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print("\n [INFO] Dang training du lieu ...")
faces,ids = getImagesandLabels(directory)
recoginzer.train(faces,np.array(ids))
recoginzer.write('/Learning/Hoc_Ki_Cuoi/Xu Ly Anh Python/Nghien Cuu Khoa Hoc/trainer/trainer.xml')

print("\n [INFO] {0} khuon mat duoc train. Thoat".format(len(np.unique(ids))))
