import operator as op
import numpy as np
import cv2
from keyImgGenerator.CAKIG import CAKIG 
from keyImgGenerator.FNCAKIG import FNCAKIG 


#################################################
if __name__ == '__main__':

        img = cv2.imread("img_baboon.png", 1)

        size = img.shape

        seed = [153128,127535,123216]
        rule = [245250,235414,10322,137962,186186,258876,204835,169565]
        dim  = 18

        param = (seed, rule, dim)
        
        img1 = CAKIG(size, param)
        img2 = FNCAKIG(size, param)
        
        img3 = np.bitwise_xor(img1,img2)
        cv2.imshow('image', img3)

        cv2.waitKey(0)
        cv2.destroyAllWindows()



#################################################





