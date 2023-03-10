import cv2
import numpy as np
from matplotlib import pyplot as plt

path = r'images/anhRonaldo.jpg'
img = cv2.imread(path , 0)
print( img.shape )
img_resize = cv2.resize(img,(320 , 400))
print( img_resize.shape )
cv2.imshow('anh goc',img)
cv2.imshow('anh resize',img_resize)


if cv2.waitKey(0) & 0xff == 27 :
    cv2.destroyAllWindows()


# hien thi histogram , anh âm bản , hàm log s