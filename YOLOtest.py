import numpy as np
import cv2
from encModule.tools.keyImgGenerator.CAKIG import CAKIG 
from encModule.tools.indexGenerator.CAIdG import CAIdG
from encModule.tools.xoring import xoring
from encModule.tools.shuffle import shuffling
from encModule.utils.ocrYOLO import ocr, full_idx, rec, roi, restore


#################################################
if __name__ == '__main__':

    plainDir  = 'img/plain/'
    resultDir = 'img/result/'
    statisDir = 'img/statistic/'

    file = 'img7.png'
    img = cv2.imread(plainDir + file)

    ############# Process 1. 영역검출 과정###############
    # input  = img
    # output = roi_img
    # YOLO를 통한 roi 검출
    # 이미지 입력시 객체별좌표와 크기 반환
    axes, size = ocr(img)
    # 객체별좌표와 크기 입력시 최대좌표 반
    idx = full_idx(axes, size)
    # 영역검출
    roi_img = roi(img,idx)


    ############# Process 2. 암호화 과정###############
    # input  = img, seed, enc=0
    # output = img
    size = roi_img.shape
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
    shuf_img = shuffling(roi_img, X, enc=0)
    enc_img = xoring(shuf_img,key_img)


    ############# Process 3. 복호화 과정###############
    DECRYPTION = 0      # 0:암호화, 1:복호화
    if DECRYPTION == 1:
        ## 복호화 실행
        nenc_img = xoring(enc_img,key_img)
        dec_img = shuffling(nenc_img, X, enc=1)
        rest_img = restore(img, dec_img, idx)
    elif DECRYPTION == 0:
        rest_img = restore(img, enc_img, idx)

    
    cv2.imshow('image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#################################################
