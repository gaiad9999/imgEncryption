import numpy as np
import cv2
from encModule.tools.keyImgGenerator.CAKIG import CAKIG 
from encModule.tools.indexGenerator.CAIdG import CAIdG
from encModule.tools.xoring import xoring
from encModule.tools.shuffle import shuffling
from encModule.utils.ocrYOLO import ocr, full_idx, rec, roi, restore



def objectDetection(img):
    ############# Process 1. 영역검출 과정###############
    # input  = img
    # output = roi_img, idx
    # YOLO를 통한 roi 검출
    # 이미지 입력시 객체별좌표와 크기 반환
    axes, size = ocr(img)
    # 객체별좌표와 크기 입력시 최대좌표 반
    idx = full_idx(axes, size)
    # 영역검출
    roi_img = roi(img,idx)

    return roi_img, idx


def roiEncryption(img, roi_img, seed, idx, enc=0):
    ############# Process 2. 암호화 과정###############
    # input  = img, roi_img, seed, idx, enc=0
    # output = img
    size = roi_img.shape
    # 파라메타 1
    dim = 18
    param1 = (245250,235414,10322,258876,204835,169565, dim)

    # 파라메타 2
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
    DECRYPTION = enc      # 0:암호화, 1:복호화
    if DECRYPTION == 1:
        ## 복호화 실행
        nenc_img = xoring(enc_img,key_img)
        dec_img = shuffling(nenc_img, X, enc=1)
        rest_img = restore(img, dec_img, idx)
    elif DECRYPTION == 0:
        rest_img = restore(img, enc_img, idx)

    return rest_img



def encryption(img, seed, idx, enc=0):
    ############# Process 2. 암호화 과정###############
    # input  = img, roi_img, seed, idx, enc=0
    # output = img
    size = img.shape
    # 파라메타 1
    dim = 18
    param1 = (245250,235414,10322,258876,204835,169565, dim)

    # 파라메타 2
    rule = (245250,235414,10322,137962,186186,258876,204835,169565)
    param2 = (seed, rule, dim)
    
    ## 암호화 과정
    # 키 좌표, 키 이미지 생성
    X = CAIdG(size, param1)
    key_img = CAKIG(size, param2)

    ## 암호화 실행
    shuf_img = shuffling(img, X, enc=0)
    img = xoring(shuf_img,key_img)


    ############# Process 3. 복호화 과정###############
    DECRYPTION = enc      # 0:암호화, 1:복호화
    if DECRYPTION == 1:
        ## 복호화 실행
        nenc_img = xoring(enc_img,key_img)
        img = shuffling(nenc_img, X, enc=1)

    return img


#################################################
if __name__ == '__main__':

    plainDir  = 'img/plain/'
    resultDir = 'img/result/'
    statisDir = 'img/statistic/'

    # 키 셋팅
    seed = (153128,127535,123216)
    #seed_od = (153127,127534,123215)


    for n in range(34,35):
        file = "img" + str(n) + ".png"
        img = cv2.imread(plainDir + file)

        # 영상처리 과정 전체
        roi_img, idx = objectDetection(img)
        rest_img = roiEncryption(img, roi_img, seed, idx, enc=0)
        enc_img = encryption(rest_img, seed, idx, enc=0)

        cv2.imwrite(resultDir+"result_"+file, enc_img)
        print("Processing "+str(n)+"...")

    
    #cv2.imshow('image', rest_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#################################################

