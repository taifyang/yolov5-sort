import cv2
import numpy as np
 
 
CLASSES=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))    
    dw, dh = (new_shape[1] - new_unpad[0])/2, (new_shape[0] - new_unpad[1])/2  # wh padding 
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im
 
 
def scale_boxes(boxes, shape, input_shape=(640, 640)):
    # Rescale boxes (xyxy) from input_shape to shape
    gain = min(input_shape[0] / shape[0], input_shape[1] / shape[1])  # gain  = old / new
    pad = (input_shape[1] - shape[1] * gain) / 2, (input_shape[0] - shape[0] * gain) / 2  # wh padding

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def draw(image, ids, confs, boxes):
    boxes = np.array(boxes)
    boxes = scale_boxes(boxes, image.shape) 
    for box, score, cl in zip(boxes, confs, ids):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

 
def infer(pred):
    confs = []
    boxes = []
    ids = []

    for i in range(pred.shape[1]):
        data = pred[0][i]
        confidence = data[4]
        if confidence > 0.6:
            score = data[5:]*confidence
            _, _, _, max_score_index = cv2.minMaxLoc(score)
            max_id = max_score_index[1]
            if score[max_id] > 0.25:
                confs.append(confidence)
                ids.append(max_id)
                x, y, w, h = data[0].item(), data[1].item(), data[2].item(), data[3].item()
                boxes.append(np.array([x-w/2, y-h/2, x+w/2, y+h/2]))
 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.25, 0.45)
    res_ids = []
    res_confs = []
    res_boxes = []
    for i in indexes:
        res_ids.append(ids[i])
        res_confs.append(confs[i])
        res_boxes.append(boxes[i])
    return res_ids, res_confs, res_boxes
 
 
if __name__=="__main__":  
    img = cv2.imread("./bus.jpg")
    net = cv2.dnn.readNet("./yolov5s.onnx")
    img_lb = letterbox(img)
    blob = cv2.dnn.blobFromImage(img_lb, 1/255., size=(640,640), swapRB=True, crop=False)
    net.setInput(blob)
    pred = net.forward()
    ids, confs, boxes = infer(pred)
    draw(img, ids, confs, boxes)
    cv2.imshow('res', img)
    cv2.waitKey()