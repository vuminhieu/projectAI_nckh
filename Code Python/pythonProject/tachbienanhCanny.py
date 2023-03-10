import cv2
import numpy as np
import matplotlib.pyplot as plt
from  scipy import ndimage

path = r'images/anhdark.jpg'
img = cv2.imread(path)
img_xam = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

h , w = img_xam.shape[:2]

kernel = np.array([[2,4,5,4,2],
           [4,9,12,9,4],
           [5,12,15,12,5],
           [4,9,12,9,4],
           [2,4,5,4,2]])

def tach_anh_Gaussian_Blur(h,w,img,kernel) :
    img_new = np.zeros([h,w])
    for i in range (2,h-2) :
        for j in range (2,w-2) :
            temp = img_xam[i-2 : i+3 , j-2 : j+3]
            img_new[i,j] = np.sum(temp*kernel) / 159
    return img_new

img_blur = tach_anh_Gaussian_Blur(h,w,img_xam,kernel)
# img_blur = cv2.GaussianBlur(img_xam,(5,5),sigmaX=1,sigmaY=1)

def tach_bien_sobel(h,w,img,maskx,masky) :
    img_new = np.zeros([h,w])
    for i in range (1,h-1) :
        for j in range (1,w-1) :
            temp = img_xam[i-1 : i+2 , j-1 : j+2]
            gx = np.sum(temp*maskx)
            gy = np.sum(temp*masky)
            img_new[i,j] = (gx ** 2 + gy ** 2) ** (1/2)
    return img_new

mask_sobel_x = np.array(([-1,0,1],
                        [-2,0,2],
                        [-1,0,1]))

mask_sobel_y = np.array(([-1,-2,-1],
                        [0,0,0],
                        [1,2,1]))

img_sobel = tach_bien_sobel(h,w,img_blur,mask_sobel_x,mask_sobel_y)

#Buoc 3 :

gx = ndimage.convolve(img_xam,mask_sobel_x)
gy = ndimage.convolve(img_xam,mask_sobel_y)
theta = np.arctan2(gy,gx)

#Non_Maxium Suppression
def non_max_suppression (img, D):
    Z = np.zeros((h, w), dtype = np.int32)
    angle = D * 180/ np.pi
    angle[angle<0] +=180

    for i in range(1, h-1):
        for j in range(1, w-1):
            #angle 0
            if (0 <= angle[i,j] <22.5) or (157.5 <= angle[i,j] <=180):
                q = img[i, j+1]
                r = img[i, j-1]

            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]

            # angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]

            # angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]

            if(img[i,j] >= q) and (img[i,j] >= r):
                Z[i,j] = img[i,j]
            else:
                Z[i,j] = 0
    return Z

img_non_max_suppression = non_max_suppression(img_sobel,theta)

plt.figure(figsize=(16,9))
plt.subplot(2,2,1), plt.imshow(img_xam,cmap='gray'), plt.title('anh xam')
plt.subplot(2,2,2), plt.imshow(img_blur,cmap='gray'), plt.title('anh blur')
plt.subplot(2,2,3), plt.imshow(img_sobel,cmap='gray'), plt.title('anh sobel')
plt.subplot(2,2,4), plt.imshow(img_non_max_suppression,cmap='gray'), plt.title('anh nms')

plt.show()

if cv2.waitKey(0) & 0xff == 27 :
    cv2.destroyAllWindows()