import cv2
import numpy as np
import matplotlib.pyplot as plt


path = r'images/anhsang.jpg'
img = cv2.imread(path)
cv2.imshow('anh goc',img)


plt.figure()


# histogram anh goc
channels = ['b','g','r']
for i,color in enumerate (channels) :
    print('vitri ' + str(i) + ' mau ' + color)
    hist_anhgoc = cv2.calcHist([img],[i],None,[256],[0,256])  #histogram anh goc
    plt.subplot(221)
    plt.title('hist anh goc')
    plt.plot(hist_anhgoc,color=color)  # show histogram anh goc


# histogram anh xam
imgxam = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hist_anhxam = cv2.calcHist([imgxam],[0],None,[256],[0,256]) #histogram anh xam
cv2.imshow('anh xam', imgxam)
plt.subplot(222)
plt.title('hist anh xam')
plt.plot(hist_anhxam) # show histogram anh xam


# anh am ban goc va hisogram
img_neg = cv2.bitwise_not(img)
cv2.imshow('anh negative' , img_neg)
# histogram anh am ban
channels = ['b','g','r']
for i,color in enumerate (channels) :
    hist_img_neg = cv2.calcHist([img_neg],[i],None,[256],[0,256])  #histogram anh am ban
    plt.subplot(223)
    plt.title('hist anh am ban')
    plt.plot(hist_img_neg,color=color)  # show histogram anh goc


# anh am ban xam va histogram
img_neg_xam = cv2.bitwise_not(imgxam)   # anh am ban xam
cv2.imshow('anh am ban xam',img_neg_xam)
# histogram anh am ban
hist_img_neg_xam = cv2.calcHist(img_neg_xam,[0],None,[256],[0,256]) # hist anh am ban xam
plt.subplot(224)
plt.title('hist anh am ban xam')
plt.plot(hist_img_neg_xam)


plt.show()


if cv2.waitKey(0) & 0xff == 27 :
    cv2.destroyAllWindows()