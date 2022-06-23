import numpy as np
import cv2
from copy import copy
import operator as op
import random as rd
from math import log10
from matplotlib import pyplot as plt


def log2(c):
    d=np.zeros(len(c))
    for i in range(len(c)):
        p=c[i]
        if p>0:
            d[i]=np.log2(p)

    return d


def dotprod(a,b):
    if len(a)!=len(b):
        print("Length ERROR")
        return
    d=0
    for i in range(len(a)):
        d+=a[i]*b[i]

    return d


def doet(img1,img2,color):
    
    img3=np.zeros((img1.shape[0],img1.shape[1]),dtype=np.uint8)
    
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i,j,color]>img2[i,j,color]:
                img3[i,j]=img1[i,j,color]-img2[i,j,color]
            else:
                img3[i,j]=img2[i,j,color]-img1[i,j,color]

    return img3



#################################################
if __name__ == '__main__':
    file1 = 'result_img33.png'
    file2 = 'result2_img33.png'
    img1 = cv2.imread(file1, 1)
    img2 = cv2.imread(file2, 1)

    

    M=img1.shape[0]
    N=img1.shape[1]
    for color in range(3):
        print(color)

        #NPCR
        c1=0
        for i in range(M):
            for j in range(N):
                if img1[i,j,color]!=img2[i,j,color]:
                    c1+=1
        #print(c4)
        c1=(c1/(M*N))*100
        #print(c4)

        #print(c5)

        print("NPCR =", end=" ")
        print(c1)


        np.seterr(over='ignore')

        
        #UACI
        #print(img1)
        #print(img2)

        img3=doet(img1,img2,color)


        #print(img3)
        c2=0
        for i in range(M):
            for j in range(N):
                c2=c2+img3[i,j]
        print(c2)
        #print(c4)
        c2=(c2/(M*N*255))*100
        #print(c4)

        #print(c5)

        print("UACI =", end=" ")
        print(c2)


        #MSE
        
        img4=img3.astype(np.int32)
        img4=np.power(img4,2)

        c3=sum(sum(img4)/M)/N
        #print(c3)
        #print(M*N)
        #c3=c3/(M*N)
        
        print("MSE =", end=" ")
        print(c3)

        #PSNR  7.7db
        c4=10*log10((255**2)/c3)
        print("PSNR =", end=" ")
        print(c4)




    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#################################################
    
