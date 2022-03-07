# _*_ coding: utf_8 _*_

import os
import cv2
import yaml
import json
import base64
import pandas as pd
import seaborn as sns
import albumentations as A
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from pathlib import Path

# ************ set randomness ************
import torch
import random
import zipfile
import os
import shutil
import xml.etree.ElementTree as ET
from numba import jit
import cv2
import numpy as np
import random
import wandb

# 동일한 input에 동일한 output이 나올 수 있도록 하는 설정
random_seed = 1656
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

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


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_bboxes(img, bboxes, classes, class_ids, colors=None, show_classes=None, bbox_format='yolo', class_name=False,
                line_thickness=2):
    image = img.copy()
    show_classes = classes if show_classes is None else show_classes
    colors = (0, 255, 0) if colors is None else colors

    if bbox_format == 'yolo':

        for idx in range(len(bboxes)):

            bbox = bboxes[idx]
            cls = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors

            if cls in show_classes:
                x1 = round(float(bbox[0]) * image.shape[1])
                y1 = round(float(bbox[1]) * image.shape[0])
                w = round(float(bbox[2]) * image.shape[1] / 2)  # w/2
                h = round(float(bbox[3]) * image.shape[0] / 2)

                voc_bbox = (x1 - w, y1 - h, x1 + w, y1 + h)
                plot_one_box(voc_bbox,
                             image,
                             color=color,
                             label=cls if class_name else str(get_label(cls)),
                             line_thickness=line_thickness)

    elif bbox_format == 'coco':

        for idx in range(len(bboxes)):

            bbox = bboxes[idx]
            cls = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors

            if cls in show_classes:
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                w = int(round(bbox[2]))
                h = int(round(bbox[3]))

                voc_bbox = (x1, y1, x1 + w, y1 + h)
                plot_one_box(voc_bbox,
                             image,
                             color=color,
                             label=cls if class_name else str(cls_id),
                             line_thickness=line_thickness)

    elif bbox_format == 'voc':

        for idx in range(len(bboxes)):

            bbox = bboxes[idx]
            cls = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors

            if cls in show_classes:
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                x2 = int(round(bbox[2]))
                y2 = int(round(bbox[3]))
                voc_bbox = (x1, y1, x2, y2)
                plot_one_box(voc_bbox,
                             image,
                             color=color,
                             label=cls if class_name else str(cls_id),
                             line_thickness=line_thickness)
    else:
        raise ValueError('wrong bbox format')

    return image

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


# Training 하고 Test 파일에 대해 zip파일을 풀어서 저정하는 부분
# 각 dir 주소 다 여기다 저장할 예정(여기만 바꾸면 ok)
main_dir = "D:/CustomData"
file_list = glob("D:/CustomData/zip_file*.zip")
os.makedirs("D:/CustomData/Train", exist_ok=True)
result_dir = "D:/CustomData/Train/"
cwd = 'D:/CustomData/file_path'

# main_dir = "G:/내 드라이브/CustomData/CustomImage/Training"
# file_list = glob("G:/내 드라이브/CustomData/CustomImage/Training/*.zip")
# file_list_img = glob("G:/내 드라이브/CustomData/CustomImage/Training/train_img/*.zip")
# file_list_xml = glob("G:/내 드라이브/CustomData/CustomImage/Training/train_xml/*.zip")
# os.makedirs("G:/내 드라이브/CustomData/CustomImage/Training/zip_xml", exist_ok=True)
# os.makedirs("G:/내 드라이브/CustomData/CustomImage/Training/zip_img", exist_ok=True)
# result_dir = "G:/내 드라이브/CustomData/CustomImage/Training/zip_xml/"
# result_img = "G:/내 드라이브/CustomData/CustomImage/Training/zip_img/"
# cwd = 'G:/내 드라이브/CustomData/CustomImage/file_path'

# for file_name in tqdm(file_list_img):
#     with zipfile.ZipFile(file_name, 'r') as zip:
#         zipInfo = zip.infolist()
#         for member in zipInfo:
#             member.filename = member.filename.encode("cp437").decode("cp949")
#             zip.extract(member,result_img)

# for file_name in tqdm(file_list_xml):
#     with zipfile.ZipFile(file_name, 'r') as zip:
#         zipInfo = zip.infolist()
#         for member in zipInfo:
#             member.filename = member.filename.encode("cp437").decode("cp949")
#             zip.extract(member,result_dir)

# # 여러 폴더 ex) 꼬깔콘, 오징어땅콩 과 같은 여러 폴더에 있는 내용을 하나의 폴더로 다 모아주는 코드 (출처 : https://gagadi.tistory.com/9)
# def read_all_file(path):
#   output = os.listdir(path)
#   file_list = []
#   for i in output:
#     if os.path.isdir(path+"/"+i):
#       file_list.extend(read_all_file(path+"/"+i))
#     elif os.path.isfile(path+"/"+i):
#       file_list.append(path+"/"+i)
#   return file_list
#
# def copy_all_file(file_list, new_path):
#   for src_path in file_list:
#     file = src_path.split("/")[-1]
#     shutil.copyfile(src_path, new_path+"/" + file)
#     print("파일 {} 작업 완료".format(file)) # 작업한 파일명 출력

#image와 label을 하나의 폴더로 합치지 않으면 이후 동작 할 수 없게 된다.
src_path = main_dir
os.makedirs(main_dir + "/Train_all",exist_ok= True)
Train_all_path = main_dir + '/Train_all'
pre_img_list = glob(result_dir + '*/*.jpg',recursive = True)
pre_label_list = glob(result_dir + '*/*.xml',recursive = True)

# src_path = main_dir
# os.makedirs(main_dir + "/Train_all",exist_ok= True)
# Train_all_path = main_dir + '/Train_all'
# pre_img_list = glob(result_img + '*/*.jpg',recursive = True)
# pre_label_list = glob(result_dir + '*/*.xml',recursive = True)
# pre_remove_img_file = glob(Train_all_path + '/*.jpg',recursive = True)
# temp_train_all_path = glob(Train_all_path + '/*.jpg',recursive = True)
# print("Damaged file remove--")
# for i in tqdm(pre_remove_img_file):
#     if os.path.exists(i):
#         os.remove(i)

print("img file copy--")

def _copyfileobj_patched(fsrc, fdst, length=16*1024*1024):
    """Patches shutil copyfileobj method to hugely improve copy speed"""
    while 1:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)

shutil.copyfileobj = _copyfileobj_patched # shutil 의 copyfileobj 대신 _copyfileobj_patched 이 호출됨

# for i in tqdm(pre_img_list):
#      if os.path.exists(i):
#         shutil.copy(i,Train_all_path)
# for i in tqdm(pre_label_list):
#     if os.path.isfile(i):
#         if not "meta" in i:
#             shutil.copy(i,Train_all_path)

# 이 부분은 나중에 고쳐야 하는 부분으로 class 이름과 class index를 추출하는 공간
# class_list는 각 index 마다의 이름 / class_index는 label로 사용될 각 class별 숫자
xml_folder_ = []
for _ ,dirs, _ in os.walk(result_dir):
    xml_folder_.extend(dirs)
random.seed(random_seed)
xml_folder_.sort()
xml_folder_ = random.sample(xml_folder_,30)
xml_folder_.sort()

class_list = []
class_index = []

for d in xml_folder_:
    class_index.append(d.split('_')[0])
    class_list.append(d.split('_')[-1])
print(len(class_list))
print(class_index)

# Yolo file train/valid/test시 쓰이게 될 hyperparameter 및 parameter 선언
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

config = AttrDict()
config.n_epoch = 10
config.batch_size = 16
config.model_fn = 'yolov5m.pt'
config.project_name = 'snack'
config.width = 640


config.height = 640
config.label_smoothing = 0.9
config.random_state = 1656
config.n_splits = 5

squeeze_xml_list = []
# Train_all 폴더에서 xml파일과 jpg 파일을 분류하는 부분
xml_list = glob(Train_all_path + "/*.xml")
# xml_list = glob("G:/내 드라이브/CustomData/CustomImage/Training/Train_all/*.xml")
for i in tqdm(xml_list):
    for j in class_index:
        if j in i and not "transformed" in i:
            squeeze_xml_list.append(i)
print(len(squeeze_xml_list))

# Data augmentation
# 데이터 증강에 필요한 transform 정의
transform = A.Compose([
    A.augmentations.crops.transforms.CenterCrop (height = 1080, width = 1080, always_apply=False, p=1.0),
    A.OneOf([
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.7),
        A.ShiftScaleRotate(p=0.7)
    ], p=1),
    # A.augmentations.crops.transforms.RandomCropNearBBox(p=1,max_part_shift=(0.3, 0.3),cropping_box_key='cropping_bbox'),
    # A.Resize(1280,1280),
    # A.RandomBrightnessContrast(p = 0.2),
], bbox_params= A.BboxParams(format = 'pascal_voc', label_fields=['category_ids']))

def Augmentation_(path,bbox,bbox_list,img_path,label_list,width_list,height_list,name,path_list,transform,idx):
    tree = ET.parse(path)
    root = tree.getroot()
    label = root.findtext("filename").split('_')[0]
    pre_trans_label = [root.findtext("filename").split('_')[0]] * len(bbox)
    pre_trans_name = root.find("object").findtext("name")
    pre_trans_img = path.split('.')[0] + '.jpg'

    category_ids = pre_trans_label

    img_array = np.fromfile(pre_trans_img, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_copy = img.copy()
    # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

    transformed = transform(image=img_copy, bboxes=bbox, category_ids=category_ids , cropping_bbox = bbox[0])
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    # transformed_category_id = transformed['category_ids']
    # print(transformed_bboxes)
    transformed_img_path = path.split('.')[0] + 'transformed' + str(idx) + '.jpg'

    # 기존 xml파일을 복사하여, 이를 수정한 후 저장해 label도 augmentation하는 작업
    source = path
    destination = transformed_img_path.split('.')[0] + '.xml'
    shutil.copyfile(source,destination)
    open_destination = open(destination, 'rt', encoding='UTF8')
    tree = ET.parse(open_destination)
    root = tree.getroot()
    root.find("filename").text = transformed_img_path
    split_path = root.find("path").text.split('/')[:]
    split_path[-1] = transformed_img_path
    root.find("path").text = '/' + os.path.join(*split_path)
    root.find("size").find("width").text = str(transformed_image.shape[0])
    root.find("size").find("height").text = str(transformed_image.shape[1])
    for i, obj in enumerate(root.iter("object")):
        if len(transformed_bboxes) > i:
            obj.find("bndbox").find("xmin").text = str(transformed_bboxes[i][0])
            obj.find("bndbox").find("ymin").text = str(transformed_bboxes[i][1])
            obj.find("bndbox").find("xmax").text = str(transformed_bboxes[i][2])
            obj.find("bndbox").find("ymax").text = str(transformed_bboxes[i][3])
        else:
            root.remove(obj)
    tree.write(destination, encoding = 'UTF-8' , xml_declaration= True)

    if len(transformed_bboxes) > 0:
    # opencv의 경우, 한글 경로는 읽고 쓰는 것이 안되기 때문에 이를 binary 형태로 encode, decode해서 저장 및 읽기를 해야한다.
    # 밑 코드는 저장 하는 코드
        extension = os.path.splitext(transformed_img_path)[1]  # 이미지 확장자
        result, encoded_img = cv2.imencode(extension, transformed_image)
        if result:
            with open(transformed_img_path, mode='w+b') as f:
                encoded_img.tofile(f)

        img_path.append(transformed_img_path)
        bbox_list.append(transformed_bboxes)
        # label_list.append(transformed_category_id)
        label_list.append(label)
        width_list.append(img.shape[1])  # img.shape -> h,w,c로 구성
        height_list.append(img.shape[0])
        name.append(pre_trans_name)
        path_list.append(destination)

    # return transformed_img_path,transformed_bboxes,transformed_category_id,img.shape,pre_trans_name,path

# Width,Height,Path,Bbox list
# width_list = []
# height_list = []
# bbox_list = []
# path_list = []
# label_list = []
# img_path = []
# name = []
# remove_path = []
# for path in tqdm(squeeze_xml_list):
#     try:
#         tree = ET.parse(path)
#     except:
#         print(path)
#         continue
#     root = tree.getroot()
#     if root.find("object") == None:
#         print(path)
#         if os.path.exists(path):
#             remove_path.append(path)
#             os.remove(Train_all_path + '/' + os.path.basename(path).split('.')[0] + '.jpg')
#             os.remove(path)
#         continue
#     if root.findtext("filename").split('_')[0] in class_index:
#         size = root.find("size")
#         width = size.findtext("width")  # size.find("width").text
#         height = size.findtext("height")
#
#         width_list.append(width)
#         height_list.append(height)
#
#         name.append(root.find("object").findtext("name"))
#         bbox = []
#         for obj in root.iter("object"):
#             xmin = obj.find("bndbox").findtext("xmin")
#             ymin = obj.find("bndbox").findtext("ymin")
#             xmax = obj.find("bndbox").findtext("xmax")
#             ymax = obj.find("bndbox").findtext("ymax")
#             bbox.append([int(xmin), int(ymin), int(xmax), int(ymax)])
#         label_list.append(root.findtext("filename").split('_')[0])
#         bbox_list.append(bbox)
#         path_list.append(path)
#         img_path.append(path.split('.')[0]+'.jpg')
#         # #Data augmentations part
#         num_of_augmentation = 10 # 각 이미지 당 augmentation 하고 싶은 개수 (ex) num_of_augmentation = 4 라면 한 이미지 당 4개가 더 늘어납니다.
#         for i in range(num_of_augmentation):
#             Augmentation_(path,bbox,bbox_list,img_path,label_list,width_list,height_list,name,path_list,transform,idx = i)
#     else:
#         continue


#img_list와 xml_list가 맞지 않는경우에 확인하는 과정 (이는 확인할 때만 주석을 해제)
# list1 = []
# list2 = []
# for i in tqdm(img_list):
#     list1.append(i.split('.')[0])
# for i in tqdm(xml_list):
#     list2.append(i.split('.')[0])
#
# print(set(list1) ^ set(list2))
# print(len(path_list))
# print(len(img_list))

# df = pd.DataFrame({
#     'img_path': img_path,  #path_list,
#     'width': width_list,
#     'height': height_list,
#     'bboxes': bbox_list,
#     'path': path_list,
# })

# # yolov5의 경우 label의 범위(숫자)가 0~len(label)로 index를 바꿔야한다. 그에 맞게 index 조절하는 코드
# print(list(set(label_list)))
# index = list(set(label_list))
# ent = {k: i for i, k in enumerate(index)}
# df['label'] = list(map(ent.get,label_list))
# index_ = list(map(str,index))
#
# print(len(df['label']))
# print("list2 after replacement is:", df['label'])
#
# df['label_path'] = df['path'].apply(lambda x: x.split('.')[0] + '.txt')
# print(len(df['label_path']))
# df['name'] = name
# print(len(df['name']))
# df = df[df['bboxes'].apply(len) > 0]  # bounding box가 없는 것은 고려하지 않는다.
# df_not_bbox = df[df['bboxes'].apply(len) <= 0]  # bounding box가 없는 list
# print(df)
# os.makedirs("D:/CustomData/result_file",exist_ok= True)
# df.to_pickle("D:/CustomData/result_file/small_test_data.pkl")
# __all__ = ['coco2yolo', 'yolo2coco', 'voc2coco', 'coco2voc', 'yolo2voc', 'voc2yolo',
#            'bbox_iou',  'load_image'] #'draw_bboxes',

df = pd.read_pickle("D:/CustomData/result_file/small_test_data.pkl")
print('read off')

from sklearn.model_selection import train_test_split, StratifiedKFold

train_path_list = None
valid_path_list = None
kfold = StratifiedKFold(n_splits=config.n_splits,
                        random_state=config.random_state,
                        shuffle=True)

for train_index, valid_index in kfold.split(X=df, y=df['label']):
    train_path_list = df.iloc[train_index]
    valid_path_list = df.iloc[valid_index]

miss_cnt = 0
all_bboxes = []
bboxes_info = []
for row_idx in tqdm(range(df.shape[0])):
    row = df.iloc[row_idx]
    image_height = int(row.height)
    image_width = int(row.width)
    bboxes_voc = np.array(row.bboxes).astype(np.float32).copy()
    num_bbox = len(bboxes_voc)
    labels = np.array([row.label] * num_bbox)[..., None].astype(str)  # [0] * 10 -> [0,0,0,0,0,0,0,0,0,0]
    # image_id = row.image_id

    ## Create Annotation(YOLO)
    with open(row.label_path, 'w') as f:
        if num_bbox < 1:
            annot = ''
            f.write(annot)
            miss_cnt += 1
            continue
        # bboxes_voc  = coco2voc(bboxes_coco, image_height, image_width)
        # bboxes_voc  = clip_bbox(bboxes_voc, image_height, image_width)
        bboxes_yolo = voc2yolo(bboxes_voc, image_height, image_width).astype(str)
        all_bboxes.extend(bboxes_yolo.astype(float))
        # bboxes_info.extend([[image_id]] * len(bboxes_yolo))

        annots = np.concatenate([labels, bboxes_yolo], axis=1)
        string = annot2str(annots)
        f.write(string)

print('Missing:', miss_cnt)

train_files = train_path_list['img_path'].values
valid_files = valid_path_list['img_path'].values

import yaml

os.makedirs(cwd, exist_ok=True)

with open(os.path.join(cwd,'train.txt'), 'w') as f:
    for path in train_files:
        f.write(path + '\n')

with open(os.path.join(cwd,'val.txt'), 'w') as f:
    for path in valid_files:
        f.write(path + '\n')

os.makedirs(cwd,exist_ok= True)

data = dict(
    path  = '',
    train = 'D:/CustomData/file_path/train.txt',
    val   = 'D:/CustomData/file_path/val.txt',
    nc    = len(index_), #  예측해야 하는 class가 2000개
    names = index_,
)

hym = dict(
  lr0= 0.01,
  lrf= 0.1,
  momentum= 0.937,
  weight_decay= 0.0005,
  warmup_epochs= 3.0,
  warmup_momentum= 0.8,
  warmup_bias_lr= 0.1,
  box= 0.05,
  cls= 0.5,
  cls_pw= 1.0,
  obj= 1.0,
  obj_pw= 1.0,
  iou_t= 0.2,
  anchor_t= 4.0,
  fl_gamma= 0.0,
  hsv_h= 0.015,
  hsv_s= 0.7,
  hsv_v= 0.4,
  degrees= 0.0,
  translate= 0.1,
  scale= 0.5,
  shear= 0.1,
  perspective= 0.0,
  flipud= 0.0,
  fliplr= 0.5,
  mosaic= 1.0,
  mixup= 0.0,
  copy_paste= 0.0,
)

with open(os.path.join(cwd,'data.yaml'), 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False,allow_unicode = True)

with open(os.path.join(cwd ,'hym.yaml'), 'w') as outfile:
    yaml.dump(hym, outfile, default_flow_style=False,allow_unicode= True)

f = open(os.path.join(cwd,'data.yaml'), 'r')
print('\nyaml:')
print(f.read())

# usage: train.py [-h] [--weights WEIGHTS] [--cfg CFG] [--data DATA] [--hyp HYP]
#                 [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--imgsz IMGSZ]
#                 [--rect] [--resume [RESUME]] [--nosave] [--noval]
#                 [--noautoanchor] [--evolve [EVOLVE]] [--bucket BUCKET]
#                 [--cache [CACHE]] [--image-weights] [--device DEVICE]
#                 [--multi-scale] [--single-cls] [--optimizer {SGD,Adam,AdamW}]
#                 [--sync-bn] [--workers WORKERS] [--project PROJECT]
#                 [--name NAME] [--exist-ok] [--quad] [--linear-lr]
#                 [--label-smoothing LABEL_SMOOTHING] [--patience PATIENCE]
#                 [--freeze FREEZE [FREEZE ...]] [--save-period SAVE_PERIOD]
#                 [--local_rank LOCAL_RANK] [--entity ENTITY]
#                 [--upload_dataset [UPLOAD_DATASET]]
#                 [--bbox_interval BBOX_INTERVAL]
#                 [--artifact_alias ARTIFACT_ALIAS]

#cwd = 'G:/내 드라이브/상품데이터셋/상품 이미지/'
# python ./Yolov5/train.py --batch 8 --imgsz 1280 --epochs 10 --data 'D:/CustomData/file_path/data.yaml' --cfg 'yolov5s.yaml' --weights 'yolov5s.pt' --save-period 1 --patience 3 --project 'convenience' --label-smoothing 0.01 --optimizer 'AdamW' --hyp 'D:/CustomData/file_path/hym.yaml'
# python ./yolov5/detect.py --img 1280 --weights C:/Users/PC/Desktop/Yolo_con/convenience/exp48/weights/best.pt --source 'D:/CustomData/small_val' --save-txt --save-conf --conf 0.03 --iou-thres 0.4 --augment