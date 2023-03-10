import cv2
import numpy as np
import matplotlib.pyplot as plt


path = r'images/anhR4.jpg'
img = cv2.imread(path)
img_xam = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


h,w = img_xam.shape[:2]

def tach_bien_robert(h,w,img,maskx,masky) :
    img_new = np.zeros([h,w])
    for i in range (1,h-1) :
        for j in range (1,w-1) :
            temp = img_xam[i-1 : i+1 , j-1 : j+1]
            gx = np.sum(temp*maskx)
            gy = np.sum(temp*masky)
            img_new[i,j] = (gx ** 2 + gy ** 2) ** (1/2)
    return img_new

mask_robert_x = np.array(([1,0],
                  [0,-1]))

mask_robert_y = np.array(([0,1],
                  [-1,0]))

img_robert = tach_bien_robert(h,w,img_xam,mask_robert_x,mask_robert_y)


def tach_bien_prewwit(h,w,img,maskx,masky) :
    img_new = np.zeros([h,w])
    for i in range (1,h-1) :
        for j in range (1,w-1) :
            temp = img_xam[i-1 : i+2 , j-1 : j+2]
            gx = np.sum(temp*maskx)
            gy = np.sum(temp*masky)
            img_new[i,j] = (gx ** 2 + gy ** 2) ** (1/2)
    return img_new

mask_prewwit_x = np.array(([-1,0,1],
                        [-1,0,1],
                        [-1,0,1]))

mask_prewwit_y = np.array(([-1,-1,-1],
                        [0,0,0],
                        [1,1,1]))

img_prewwit = tach_bien_prewwit(h,w,img_xam,mask_prewwit_x,mask_prewwit_y)


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

img_sobel = tach_bien_sobel(h,w,img_xam,mask_sobel_x,mask_sobel_y)


# def tach_bien_kirsch(h,w,img,mask) :
#     sum_array = []
#     for i in range (1,h-1) :
#         for j in range (1,w-1) :
#             for m in range(len(mask)) :
#                 temp = img[ i-1 : i+2 , j-1 : j+2 ]
#                 sum_array.append(np.sum(temp*mask[m]))
#             img[i,j] = max(sum_array)
#             sum_array.clear()
#     return img
#
# mask_kirsch_1 = np.array(([5,5,5],
#                         [-3,0,-3],
#                         [-3,-3,-3]))
#
# mask_kirsch_2 = np.array(([-3,5,5],
#                         [-3,0,5],
#                         [-3,-3,-3]))
#
# mask_kirsch_3 = np.array(([-3,-3,5],
#                         [-3,0,5],
#                         [-3,-3,5]))
#
# mask_kirsch_4 = np.array(([-3,-3,-3],
#                         [-3,0,5],
#                         [-3,5,5]))
#
# mask_kirsch_5 = np.array(([-3,-3,-3],
#                         [-3,0,-3],
#                         [5,5,5]))
#
# mask_kirsch_6 = np.array(([-3,-3,-3],
#                         [5,0,-3],
#                         [5,5,-3]))
#
# mask_kirsch_7 = np.array(([5,-3,-3],
#                         [5,0,-3],
#                         [5,-3,-3]))
#
# mask_kirsch_8 = np.array(([5,5,-3],
#                         [5,0,-3],
#                         [-3,-3,-3]))
#
# mask_kirsch = [mask_kirsch_1,mask_kirsch_2,mask_kirsch_3,mask_kirsch_4,mask_kirsch_5,mask_kirsch_6,mask_kirsch_7,mask_kirsch_8]
# img_kirsch = tach_bien_kirsch(h,w,img_xam,mask_kirsch)

plt.figure(figsize=(16,9))
plt.subplot(2,2,1), plt.imshow(img_xam,cmap='gray'), plt.title('anh xam')
plt.subplot(2,2,2), plt.imshow(img_robert,cmap='gray'), plt.title('anh robert')
plt.subplot(2,2,3), plt.imshow(img_prewwit,cmap='gray'), plt.title('anh prewwit')
plt.subplot(2,2,4), plt.imshow(img_sobel,cmap='gray'), plt.title('anh sobel')
# plt.subplot(2,3,5), plt.imshow(img_kirsch,cmap='gray'), plt.title('anh kirsch')

plt.show()

