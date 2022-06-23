import numpy as np
import operator as op
import cv2
import time
from copy import copy
import scipy.stats as ss


#고속 CA셔플링


#################################################
#CA 함수
def CA8(data,rule=15):
    return ((data&rule)^(data//2)^(data*2))%256


def CA(data,rule=179,dim=8):
    return ((data&rule)^(data//2)^(data*2))%(2**dim)


def FNCA(data,rule=245250,dim=18):
    return ((data&rule)^(data//2)^(data//4)^(data*4)^(data*2))%(2**dim)

def CArot(data,ruleList,dim):
    for rule in ruleList:
        data=CA(data,rule,dim)
    return data


# 배열 정렬 함수
def scasort(data):
    srt=ss.rankdata(data)-1
    return srt.astype(np.uint)


## CA 셔플링을 위한 CA처리가 12번 일어난다.
# 여기도 전체적으로 좀 수정해야할듯
def CAIdG(size, param):

    sn1, sn2, channel = size
    r1, r2, r3, r4, r5, r6, dim = param

    x1=np.arange(sn1)
    x2=np.arange(sn2)

    CArule1=[r1,r2,r3,r4,r5,r6,r1,r2,r3,r4,r5,r6]
    CArule2=[r3,r4,r5,r6,r1,r2,r3,r4,r5,r6,r1,r2]
    CArule3=[r5,r6,r1,r2,r3,r4,r5,r6,r1,r2,r3,r4]
    # 
    x11=scasort(CArot(x1,CArule1,dim))
    x12=scasort(CArot(x11,CArule2,dim))
    x13=scasort(CArot(x12,CArule3,dim))
    
    x21=scasort(CArot(x2,CArule1,dim))
    x22=scasort(CArot(x21,CArule2,dim))
    x23=scasort(CArot(x22,CArule3,dim))


    X = (x11, x12, x13, x21, x22, x23)


    return X




#################################################
if __name__ == '__main__':

    ##### 아래는 좌표행렬 #####
    ##
    ## x=[0,1,...,254]
    ## x+1=[1,2,...,255]
    ## CA(x)는 x에 8셀 CA를 취하는 함수
    ## sn1 : 높이 // sn2 : 너비

    size = (255, 255, 3)
    
    r1=245250
    r2=235414
    r3=10322
    r4=258876
    r5=204835
    r6=169565

    dim=18
    param = (r1, r2, r3, r4, r5, r6, dim)

    X = CAIdG(size, param)
    
    
#################################################
