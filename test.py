import numpy as np
import cv2
from encModule.tools.keyImgGenerator.CAKIG import CAKIG 
from encModule.tools.indexGenerator.CAIdG import CAIdG
from encModule.tools.xoring import xoring
from encModule.tools.shuffle import shuffling

#################################################
if __name__ == '__main__':
    
    plainDir = 'img/plain/'
    file = 'img_baboon.png'
    img = cv2.imread(plainDir + file)

    size = img.shape
    # 파라메타 1
    dim = 18
    param1 = (245250,235414,10322,258876,204835,169565, dim)

    # 파라메타 2
    seed = (153128,127535,123216)
    rule = (245250,235414,10322,137962,186186,258876,204835,169565)
    param2 = (seed, rule, dim)
    
    ## 암호화 과정
    # 키 좌표, 키 이미지 생성
    X = CAIdG(size, param1)
    key_img = CAKIG(size, param2)

    ## 암호화 실행
    shuf_img = shuffling(img, X, enc=0)
    enc_img = xoring(shuf_img,key_img)

    DECRYPTION = 1
    if DECRYPTION == 1:
        ## 복호화 실행
        nenc_img = xoring(enc_img,key_img)
        img = shuffling(nenc_img, X, enc=1)
    elif DECRYPTION == 0:
        img = enc_img

    cv2.imshow('image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#################################################
