import cv2
import numpy as np
from matplotlib import pyplot as plt

path = r'images/anhR4.jpg'
img = cv2.imread(path)

B = img[:,:,0]
G = img[:,:,1]
R = img[:,:,2]

img_xam = 0.2989 * R + 0.5870 * G + 0.1140 * B

plt.figure()
plt.subplot(1,1,1), plt.imshow(img_xam,cmap='gray'), plt.title('anh xam')
plt.show()

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


