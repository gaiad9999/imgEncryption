import numpy as np
import cv2


# Loading image
def ocr(img):
    
    height, width, channels = img.shape

    # Load Yolo
    net = cv2.dnn.readNet("encModule/utils/parameters/yolov3.weights", "encModule/utils/parameters/yolov3.cfg")
    classes = []
    with open("encModule/utils/parameters/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 객체 검출
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    #class_ids = []
    #confidences = []
    boxes = []
    axes = [[],[],[],[]]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            ## class_id 최대값의 위치가 0(=인간)일때만 동작
            class_id = np.argmax(scores)
            if class_id == 0:
                ## 유의수준이 50%보다 클 때만 동작
                confidence = scores[class_id]
                if confidence > 0.5:
                    # 객체별 유의수준 및 아이디
                    #confidences.append(float(confidence))
                    #class_ids.append(class_id)
                    # 좌표값 추출
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # 좌표도출 : 객체별
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    # 좌표도출 : 좌표축별
                    x1 = int(center_x - w / 2)
                    axes[0].append(x1)
                    x2 = int(center_x + w / 2)
                    axes[1].append(x2)
                    y1 = int(center_y - h / 2)
                    axes[2].append(y1)
                    y2 = int(center_y + h / 2)
                    axes[3].append(y2)
    #print(boxes)
    #print(axes)

    return axes, (height, width)


def full_idx(axes, size):
    ##indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    ##for i in range(len(boxes)):
    ##    if i in indexes:
    ##        x, y, w, h = boxes[i]
    ##        color = colors[i]
    ##        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    height, width = size

    x_min = min(axes[0])
    if x_min < 0: x_min = 0
    x_max = max(axes[1])
    if x_max > width: x_max = width
    y_min = min(axes[2])
    if y_min < 0: y_min = 0
    y_max = max(axes[3])
    if y_max > height: y_max = height

    return (x_min, x_max, y_min, y_max)


def rec(img, idx):
    x_min, x_max, y_min, y_max = idx
    color = np.random.uniform(0, 255)
    return cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

def roi(img, idx):
    x_min, x_max, y_min, y_max = idx
    return img[y_min:y_max, x_min:x_max]

def restore(img, roi, idx):
    x_min, x_max, y_min, y_max = idx
    img[y_min:y_max, x_min:x_max] = roi
    return img


#################################################
if __name__ == '__main__':

    img_name = "img7.png"
    img = cv2.imread(img_name)

    # 이미지 입력시 객체별좌표와 크기 반환
    axes, size = ocr(img)
    # 객체별좌표와 크기 입력시 최대좌표 반
    idx = full_idx(axes, size)
    # 이미지에 사각형 표시
    img = rec(img, idx)

    # 영역검출
    roi_img = roi(img,idx)

    cv2.imshow("Image", roi_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 검출영역 복원
    rest_img = restore(img,roi_img,idx)
    
    cv2.imshow("Image", rest_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#################################################
