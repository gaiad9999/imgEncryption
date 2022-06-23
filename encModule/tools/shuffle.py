import numpy as np
import cv2
from copy import copy

#고속 셔플링


#################################################

def shuffling(img, X, enc=0):
    ### 아놀드 캣맵을 이미지에 n회 적용. N=255인 경우 주기=90
    
    ### a가 0이면 암호화 1이면 복호화 ###
    
    a=enc

    x11, x12, x13, x21, x22, x23 = X

    #####################################

    width, height, channel = img.shape
    x1=np.arange(width)
    x2=np.arange(height)

    if a==0:
        img1=copy(img)
        for i in range(width):
            img1[i,:,0]=img[x11[i],:,0]
            img1[i,:,1]=img[x12[i],:,1]
            img1[i,:,2]=img[x13[i],:,2]

        img2=copy(img1)
        for i in range(height):
            img2[:,i,0]=img1[:,x21[i],0]
            img2[:,i,1]=img1[:,x22[i],1]
            img2[:,i,2]=img1[:,x23[i],2]

    #start_time = time.time()

    ### 이하 복호화 파트
    ### 복호화 CA셔플링

    if a==1:
        img1=copy(img)
        for i in range(height):
            img1[:,x21[i],0]=img[:,i,0]
            img1[:,x22[i],1]=img[:,i,1]
            img1[:,x23[i],2]=img[:,i,2]
            
        img2=copy(img1)
        for i in range(width):
            img2[x11[i],:,0]=img1[i,:,0]
            img2[x12[i],:,1]=img1[i,:,1]
            img2[x13[i],:,2]=img1[i,:,2]
    
    #if np.equal(img,img1):
    #    print("True")
    return img2




#################################################
if __name__ == '__main__':
    from indexGenerator.CAIdG import CAIdG

    file = 'img_baboon.png'
    img = cv2.imread(file, 1)

    ##### 아래는 좌표행렬 #####
    ##
    ## x=[0,1,...,254]
    ## x+1=[1,2,...,255]
    ## CA(x)는 x에 8셀 CA를 취하는 함수
    ## sn1 : 높이 // sn2 : 너비
    
    r1  = 245250
    r2  = 235414
    r3  = 10322
    r4  = 258876
    r5  = 204835
    r6  = 169565
    dim = 18

    param = (r1, r2, r3, r4, r5, r6, dim)

    X = CAIdG(img.shape, param)
    img2 = shuffling(img, X, enc=0)

    img3 = shuffling(img2, X, enc=1)

    cv2.imshow('image', img3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#################################################
