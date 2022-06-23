import time
import numpy as np
import operator as op
import cv2


# CA로 초고속 키이미지 생성
# CA key image generator

def CAKIG(size, param):
        """
        CAKIG
        - CA key image generator
        Input  : size = ( width:int,
                          height:int,
                          channel:int )
                 param = ( seed:list[3],
                           rule:list[8],
                           dim :int )
        Output : seq = list[size]

        - Output 값은 Key 이미지이다.
        * rule에 대한 이슈
          - 현재 여기서 rule은 각 CA seqing마다 다른 룰이 사용되며 총 6개를 사용함.
          - rule을 조금 더 효율적으로 컨트롤 할 방법을 강구해보자.
        """
        ## 1. Init
        size1, size2, channel = size
        seed = param[0]
        rule = param[1]
        dim = param[2]

        d = np.zeros((size1,size2,channel),dtype=np.uint)

        i=0

        ## 2. Key 수열 생성
        for z in range(3):
                x=seed[z]
                for j in range(d.shape[1]):
                        k=np.bitwise_xor(np.right_shift(x,1),np.left_shift(x,1))
                        r=np.bitwise_and(x,rule[z])
                        d[i,j,z]=np.bitwise_xor(k,r)%(2**dim)
                        x=d[i,j,z]

                x=d[i,:,z]
                for i in range(d.shape[0]):
                        k=np.bitwise_xor(np.right_shift(x,1),np.left_shift(x,1))
                        r=np.bitwise_and(x,rule[z+3])
                        d[i,:,z]=np.bitwise_xor(k,r)%(2**dim)
                        x=d[i,:,z]


        ## 3. Key 수열을 Key 이미지로 변환
        keyImg = np.zeros((size1,size2,3),dtype=np.uint8)
        # issue. 이걸 for문 사용하지 않고 구현하는 방법은?
        for i in range(size1):
                for j in range(size2):
                        keyImg[i,j]=d[i,j]%256

        ## 4. 출력
        return keyImg


#################################################
if __name__ == '__main__':

        img = cv2.imread("img_baboon.png", 1)

        size = img.shape

        seed = [153128,127535,123216]
        rule = [245250,235414,10322,137962,186186,258876,204835,169565]
        dim  = 18

        param = (seed, rule, dim)
        
        img1 = CAKIG(size, param)
        
        img3 = np.bitwise_xor(img1,img)
        cv2.imshow('image', img3)

        cv2.waitKey(0)
        cv2.destroyAllWindows()



        ##CA파트 보강설명
        ##아래의 과정을 전부 합친것이 위의 CA출력값인 d
        ##c=np.bitwise_xor(np.right_shift(x,1),np.left_shift(x,1))
        ##e=np.bitwise_and(x,rule)
        ##d=np.bitwise_xor(c,e)
#################################################



