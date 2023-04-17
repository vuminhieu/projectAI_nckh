import cv2
import numpy as np
from matplotlib import pyplot as plt


path = r'images/anhR4.jpg'
img = cv2.imread(path)
img_xam = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
column,row = img_xam.shape[:2]

# them nhieu Gauss vao anh
loc = 10
scale = 20
noise_gauss = np.random.normal(loc,scale,size=(column,row))
img_Gauss = img_xam + noise_gauss

#them nhieu muoi tieu den :
number_black = int(column*row*0.02)
number_white = int(column*row*0.02)
column_blacks = np.random.randint(0,column,number_black)
row_blacks = np.random.randint(0,row,number_black)
column_whites = np.random.randint(0,column,number_white)
row_whites = np.random.randint(0,row,number_white)
img_SP_noise = np.copy(img_xam)
img_SP_noise[column_blacks,row_blacks] = 0
img_SP_noise[column_whites,row_whites] = 255

#them nhieu luong tu hoa :
a , b = 1 , 50
noise_uniform = np.random.uniform(low=a , high=b , size=(column,row))
img_Uni_Noise = img_xam + noise_uniform

#them nhieu poisson :
lam = 100
noise_poisson = np.random.poisson(lam , size=(column,row))
img_Poisson_Noise = img_xam + noise_poisson

# tao cua so figure các anh nhieu
plt.figure(figsize=(16,9))
#hien thi anh goc xam
plt.subplot(2,3,1)
plt.imshow(img_xam,cmap='gray')
plt.title('anh xam')
# hien thi anh bi nhieu Gauss
plt.subplot(2,3,2).imshow(img_Gauss,cmap='gray')
plt.title('anh Gauss')
# #hien thi anh bi nhieu SP
plt.subplot(2,3,3).imshow(img_SP_noise,cmap='gray')
plt.title('anh SP')
#hien thi anh bi nhieu Luong tu hoa
plt.subplot(2,3,4).imshow(img_Uni_Noise,cmap='gray')
plt.title('anh Uni')
#hien thi anh bi nhieu Poisson
plt.subplot(2,3,5).imshow(img_Poisson_Noise,cmap='gray')
plt.title('anh Poisson')


#loc anh bi nhieu :

# a. Loc anh bang phg phap loc Trung Binh
# Cach 1 ( tinh toan ) :
# kernel1 = np.ones((3, 3), np.float32) / 9
kernel1 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]]) / 9
img_LocTB = cv2.filter2D(src= img_SP_noise, ddepth=-1, kernel=kernel1)
# Cach 2 ( tvien co san ) :
# img_LocTB = cv2.blur(img_SP_noise , ksize=(3,3))

#b. Loc anh bang phg phap Loc Trung Vi
img_LocTVi = cv2.medianBlur(img_SP_noise , ksize= 5 )

#c. Loc anh bang phg phap sac net
kernel3 = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
img_Sacnet = cv2.filter2D(src=img_LocTVi, ddepth=-1, kernel=kernel3)

# tao cua so figure voi bo loc nhieu
figure2 = plt.figure(figsize=(16,9))
(ax1,ax2),(ax3,ax4) = figure2.subplots(2,2)
#hien thi anh bi nhieu
ax1.imshow(img_SP_noise,cmap='gray')
ax1.set_title('anh SP')
#hien thi anh loc trung binh nhieu
ax2.imshow(img_LocTB,cmap='gray')
ax2.set_title('anh loc trung binh nhieu ')
#hien thi anh loc trung vi nhieu
ax3.imshow(img_LocTVi , cmap='gray')
ax3.set_title('anh loc trung vi nhieu ')
#hien thi anh loc sac net
ax4.imshow(img_Sacnet,cmap='gray')
ax4.set_title('anh loc sac net')

plt.show()

if cv2.waitKey(0) & 0xff == 27 :
    cv2.destroyAllWindows()


