from numba import jit
import numpy as np
import cv2

# ************ Yolov5 Preprocessing module ************

# 밑의 코드들은 각각 다른 annotation의 표기에 따라 자신이 원하는 형식으로 bounding box를 변경하는 코드

@jit(nopython=True)
def voc2yolo(bboxes, height=720, width=1280):
    """
    voc  => [x1, y1, x2, y1]
    yolo => [xmid, ymid, w, h] (normalized)
    """

    #     bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., 0::2] /= width
    bboxes[..., 1::2] /= height

    bboxes[..., 2] -= bboxes[..., 0]
    bboxes[..., 3] -= bboxes[..., 1]

    bboxes[..., 0] += bboxes[..., 2] / 2
    bboxes[..., 1] += bboxes[..., 3] / 2

    return bboxes


@jit(nopython=True)
def yolo2voc(bboxes, height=720, width=1280):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]

    """
    #     bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., 0::2] *= width
    bboxes[..., 1::2] *= height

    bboxes[..., 0:2] -= bboxes[..., 2:4] / 2
    bboxes[..., 2:4] += bboxes[..., 0:2]

    return bboxes


@jit(nopython=True)
def coco2yolo(bboxes, height=720, width=1280):
    """
    coco => [xmin, ymin, w, h]
    yolo => [xmid, ymid, w, h] (normalized)
    """

    #     bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int

    # normolizinig
    bboxes[..., 0::2] /= width
    bboxes[..., 1::2] /= height

    # converstion (xmin, ymin) => (xmid, ymid)
    bboxes[..., 0:2] += bboxes[..., 2:4] / 2

    return bboxes


@jit(nopython=True)
def yolo2coco(bboxes, height=720, width=1280):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    coco => [xmin, ymin, w, h]

    """
    #     bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int

    # denormalizing
    bboxes[..., 0::2] *= width
    bboxes[..., 1::2] *= height

    # converstion (xmid, ymid) => (xmin, ymin)
    bboxes[..., 0:2] -= bboxes[..., 2:4] / 2

    return bboxes


@jit(nopython=True)
def voc2coco(bboxes, height=720, width=1280):
    """
    voc  => [xmin, ymin, xmax, ymax]
    coco => [xmin, ymin, w, h]

    """
    #     bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int

    # converstion (xmax, ymax) => (w, h)
    bboxes[..., 2:4] -= bboxes[..., 0:2]

    return bboxes


@jit(nopython=True)
def coco2voc(bboxes, height=720, width=1280):
    """
    coco => [xmin, ymin, w, h]
    voc  => [xmin, ymin, xmax, ymax]

    """
    #     bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int

    # converstion (w, h) => (w, h)
    bboxes[..., 2:4] += bboxes[..., 0:2]

    return bboxes


# 밑의 코드는 bounding box의 IOU(겹치는 영역)을 계산해주는 코드이다.

@jit(nopython=True)
def bbox_iou(b1, b2):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    Args:
        b1 (np.ndarray): An ndarray containing N(x4) bounding boxes of shape (N, 4) in [xmin, ymin, xmax, ymax] format.
        b2 (np.ndarray): An ndarray containing M(x4) bounding boxes of shape (N, 4) in [xmin, ymin, xmax, ymax] format.

    Returns:
        np.ndarray: An ndarray containing the IoUs of shape (N, 1)
    """
    #     0 = np.convert_to_tensor(0.0, b1.dtype)
    # b1 = b1.astype(np.float32)
    # b2 = b2.astype(np.float32)
    b1_xmin, b1_ymin, b1_xmax, b1_ymax = np.split(b1, 4, axis=-1)
    b2_xmin, b2_ymin, b2_xmax, b2_ymax = np.split(b2, 4, axis=-1)
    b1_height = np.maximum(0, b1_ymax - b1_ymin)
    b1_width = np.maximum(0, b1_xmax - b1_xmin)
    b2_height = np.maximum(0, b2_ymax - b2_ymin)
    b2_width = np.maximum(0, b2_xmax - b2_xmin)
    b1_area = b1_height * b1_width
    b2_area = b2_height * b2_width

    intersect_xmin = np.maximum(b1_xmin, b2_xmin)
    intersect_ymin = np.maximum(b1_ymin, b2_ymin)
    intersect_xmax = np.minimum(b1_xmax, b2_xmax)
    intersect_ymax = np.minimum(b1_ymax, b2_ymax)
    intersect_height = np.maximum(0, intersect_ymax - intersect_ymin)
    intersect_width = np.maximum(0, intersect_xmax - intersect_xmin)
    intersect_area = intersect_height * intersect_width

    union_area = b1_area + b2_area - intersect_area
    iou = np.nan_to_num(intersect_area / union_area).squeeze()

    return iou


# 밑의 코드는 bounding box를 그려내는 코드 (사실상 Yolov5 module 안에서 이미 그려주고 있어 예비용으로 넣어 두었다.)

@jit(nopython=True)
def clip_bbox(bboxes_voc, height=720, width=1280):
    """Clip bounding boxes to image boundaries.

    Args:
        bboxes_voc (np.ndarray): bboxes in [xmin, ymin, xmax, ymax] format.
        height (int, optional): height of bbox. Defaults to 720.
        width (int, optional): width of bbox. Defaults to 1280.

    Returns:
        np.ndarray : clipped bboxes in [xmin, ymin, xmax, ymax] format.
    """
    bboxes_voc[..., 0::2] = bboxes_voc[..., 0::2].clip(0, width)
    bboxes_voc[..., 1::2] = bboxes_voc[..., 1::2].clip(0, height)
    return bboxes_voc


# 밑의 코드들은 annotation에서 string으로 혹은 string에서 annotation으로 변환시켜주는 코드이다.

def str2annot(data):
    """Generate annotation from string.

    Args:
        data (str): string of annotation.

    Returns:
        np.ndarray: annotation in array format.
    """
    data = data.replace('\n', ' ')
    data = np.array(data.split(' '))
    annot = data.astype(float).reshape(-1, 5)
    return annot


def annot2str(data):
    """Generate string from annotation.

    Args:
        data (np.ndarray): annotation in array format.

    Returns:
        str: annotation in string format.
    """
    data = data.astype(str)
    string = '\n'.join([' '.join(annot) for annot in data])
    return string

def load_image(image_path):
    return cv2.imread(image_path)[..., ::-1]