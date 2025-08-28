import cv2
import numpy as np
from copy import copy


def filter_boxes(boxes, box_confidences, box_class_probs, OBJ_THRESH):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores, NMS_THRESH):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def dfl(postion):
    n, c, h, w = postion.shape
    p_num = 4
    mc = c // p_num
    y = postion.reshape(n, p_num, mc, h, w)
    y = softmax(y, 2)
    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y
    

def softmax(data, dim):
    max = np.max(data, axis=dim, keepdims=True).repeat(data.shape[dim], axis=dim)
    exps = np.exp(data - max)
    return exps / np.sum(exps, axis=dim, keepdims=True)


def box_process(position, IMG_SIZE):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy


def post_process(input_data, OBJ_THRESH, NMS_THRESH, IMG_SIZE):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i], IMG_SIZE))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf, OBJ_THRESH)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s, NMS_THRESH)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def letter_box(im, new_shape, color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def get_real_box(src_shape, box, dw, dh, ratio):
    bbox = copy(box)
    # unletter_box result
    bbox[:,0] -= dw
    bbox[:,0] /= ratio
    bbox[:,0] = np.clip(bbox[:,0], 0, src_shape[1])

    bbox[:,1] -= dh
    bbox[:,1] /= ratio
    bbox[:,1] = np.clip(bbox[:,1], 0, src_shape[0])

    bbox[:,2] -= dw
    bbox[:,2] /= ratio
    bbox[:,2] = np.clip(bbox[:,2], 0, src_shape[1])

    bbox[:,3] -= dh
    bbox[:,3] /= ratio
    bbox[:,3] = np.clip(bbox[:,3], 0, src_shape[0])
    return bbox


def yolov8_infer(img_src, rknn_lite, IMG_SIZE, OBJ_THRESH, NMS_THRESH):
    src_shape = img_src.shape[:2]
    img, ratio, (dw, dh) = letter_box(img_src, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input = np.expand_dims(img, axis=0)
    outputs = rknn_lite.inference(inputs=[input])
    boxes, classes, scores = post_process(outputs, OBJ_THRESH, NMS_THRESH, IMG_SIZE)
    if boxes is not None:
        boxes = get_real_box(src_shape, boxes, dw, dh, ratio)
        return boxes.astype(np.int32).tolist(), classes.astype(np.int32).tolist(), scores.astype(np.float32).tolist()
    else:
        return [], [], []


def infer_single_image(args):
    img_src, rknn_lite, IMG_SIZE, OBJ_THRESH, NMS_THRESH = args
    return yolov8_infer(img_src, rknn_lite, IMG_SIZE, OBJ_THRESH, NMS_THRESH)
