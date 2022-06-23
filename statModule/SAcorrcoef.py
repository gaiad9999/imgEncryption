import numpy as np
import cv2
from copy import copy
import operator as op
import random as rd
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


#################################################
if __name__ == '__main__':
    file = 'img33.png'
    img = cv2.imread(file, 1)


    M=img.shape[0]
    N=img.shape[1]
    corr=[]

    for color in range(3):
        print(color)
        
        a=2000
        t=300



        #가로쌍 상관계수
        rod=[]

        for k in range(t):

            #좌표추출
            num=[]
            for i in range(254):
                for j in range(254):
                    txt=i,j
                    num.append(txt)
        
            rd.shuffle(num)
            num=num[0:a]

            #픽셀값추출
            c1=np.zeros([2,a])
            for i in range(a):
                c1[0,i]=img[num[i][0],num[i][1],color]
                c1[1,i]=img[num[i][0]+1,num[i][1],color]

            cor=np.corrcoef(c1[0],c1[1])[1,0]
            rod.append(cor)

        u=sum(rod)/t
        corr.append(u)
        
        print("Horizontal =", end=" ")
        print(u)




        #세로쌍 상관계수
        rod=[]

        for k in range(t):

            #좌표추출
            num=[]
            for i in range(254):
                for j in range(254):
                    txt=i,j
                    num.append(txt)
        
            rd.shuffle(num)
            num=num[0:a]

            #픽셀값추출
            c2=np.zeros([2,a])
            for i in range(a):
                c2[0,i]=img[num[i][0],num[i][1],color]
                c2[1,i]=img[num[i][0],num[i][1]+1,color]

            cor=np.corrcoef(c2[0],c2[1])[1,0]
            rod.append(cor)

        u=sum(rod)/t
        corr.append(u)

        print("Vertical =", end=" ")
        print(u)



        
        #대각쌍 상관계수
        rod=[]

        for k in range(t):

            #좌표추출
            num=[]
            for i in range(254):
                for j in range(254):
                    txt=i,j
                    num.append(txt)
        
            rd.shuffle(num)
            num=num[0:a]

            #픽셀값추출
            c3=np.zeros([2,a])
            for i in range(a):
                c3[0,i]=img[num[i][0],num[i][1],color]
                c3[1,i]=img[num[i][0]+1,num[i][1]+1,color]

            cor=np.corrcoef(c3[0],c3[1])[1,0]
            rod.append(cor)

        u=sum(rod)/t
        corr.append(u)

        print("Diagonal =", end=" ")
        print(u)

print("Average")
print("Horizontal =", end=" ")
print((corr[0]+corr[3]+corr[6])/3)
print("Vertical =", end=" ")
print((corr[1]+corr[4]+corr[7])/3)
print("Diagonal =", end=" ")
print((corr[2]+corr[5]+corr[8])/3)



    
#################################################
    
