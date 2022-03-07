# _*_ coding: utf_8 _*_

# ************ import module ************

import os
import cv2
import yaml
import json
import pandas as pd
import albumentations as A
from glob import glob
from tqdm import tqdm
import os
import shutil
import xml.etree.ElementTree as ET
from numba import jit
import cv2
import numpy as np
import random
import wandb
import torch
import random

from Yolov5_preprocessing import voc2yolo,yolo2voc,coco2yolo,yolo2coco,voc2coco,coco2voc,bbox_iou,clip_bbox,str2annot,annot2str,load_image
from File_Tool import zip_file_extract,copy_img_label
from Data_Augmentation import Augmentation_

# ************ set randomness ************

# 매번 code를 실행할 때마다 같은 결과가 나올 수 있도록 하는 코드

random_seed = 1656
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

# directory address
main_dir = "D:/CustomData"
file_list = glob("D:/CustomData/zip_file*.zip")
os.makedirs("D:/CustomData/Train", exist_ok=True)
result_dir = "D:/CustomData/Train/"
cwd = 'D:/CustomData/file_path'

# ************ zip file extract ************
# print("zip file extract")
# zip_file_extract(file_list,result_dir)

# Combine the images and labels in one folder
os.makedirs(main_dir + "/Train_all",exist_ok= True)
Train_all_path = main_dir + '/Train_all'
pre_img_list = glob(result_dir + '*/*.jpg',recursive = True)
pre_label_list = glob(result_dir + '*/*.xml',recursive = True)

# ************ Image Label Combine ************
# print("img file copy")
# copy_img_label(pre_img_list,pre_label_list,Train_all_path)

# class_list는 각 index 마다의 이름 / class_index는 label로 사용될 각 class별 숫자

xml_folder_ = []
for _ ,dirs, _ in os.walk(result_dir):
    xml_folder_.extend(dirs)
random.seed(random_seed)
xml_folder_.sort()
xml_folder_ = random.sample(xml_folder_,200)
xml_folder_.sort()

class_list = []
class_index = []
for d in xml_folder_:
    class_index.append(d.split('_')[0])
    class_list.append(d.split('_')[-1])
print(len(class_list))
print(class_index)


# ************ Define config ************
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


# Train_all 폴더에서 이전에 transform된 xml파일들을 반영하지 않도록 만드는 코드
squeeze_xml_list = []
xml_list = glob(Train_all_path + "/*.xml")
for i in tqdm(xml_list):
    for j in class_index:
        if j in i and not "transformed" in i:
            squeeze_xml_list.append(i)
print(len(squeeze_xml_list))


# ************ Preprocessing for DataFrame ************

# Width,Height,Path,Bbox list
width_list = []
height_list = []
bbox_list = []
path_list = []
label_list = []
img_path = []
name = []
remove_path = []
for path in tqdm(squeeze_xml_list):
    try:
        tree = ET.parse(path)
    except:
        print(path)
        continue
    root = tree.getroot()
    if root.find("object") == None:
        print(path)
        if os.path.exists(path):
            remove_path.append(path)
            os.remove(Train_all_path + '/' + os.path.basename(path).split('.')[0] + '.jpg')
            os.remove(path)
        continue
    if root.findtext("filename").split('_')[0] in class_index:
        size = root.find("size")
        width = size.findtext("width")  # size.find("width").text
        height = size.findtext("height")

        width_list.append(width)
        height_list.append(height)

        name.append(root.find("object").findtext("name"))
        bbox = []
        for obj in root.iter("object"):
            xmin = obj.find("bndbox").findtext("xmin")
            ymin = obj.find("bndbox").findtext("ymin")
            xmax = obj.find("bndbox").findtext("xmax")
            ymax = obj.find("bndbox").findtext("ymax")
            bbox.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        label_list.append(root.findtext("filename").split('_')[0])
        bbox_list.append(bbox)
        path_list.append(path)
        img_path.append(path.split('.')[0]+'.jpg')

        # #Data augmentations part
        num_of_augmentation = 10 # 각 이미지 당 augmentation 하고 싶은 개수 (ex) num_of_augmentation = 4 라면 한 이미지 당 4개가 더 늘어납니다.
        for i in range(num_of_augmentation):
            Augmentation_(path,bbox,bbox_list,img_path,label_list,width_list,height_list,name,path_list,idx = i)
    else:
        continue


# ************ Create DataFrame ************

df = pd.DataFrame({
    'img_path': img_path,  #path_list,
    'width': width_list,
    'height': height_list,
    'bboxes': bbox_list,
    'path': path_list,
})

# yolov5의 경우 label의 범위(숫자)가 0~len(label)로 index를 바꿔야한다. 그에 맞게 index 조절하는 코드
index = list(set(label_list))
ent = {k: i for i, k in enumerate(index)}
df['label'] = list(map(ent.get,label_list))
index_ = list(map(str,index))

df['label_path'] = df['path'].apply(lambda x: x.split('.')[0] + '.txt')
df['name'] = name
df = df[df['bboxes'].apply(len) > 0]  # bounding box가 없는 것은 고려하지 않는다.

__all__ = ['coco2yolo', 'yolo2coco', 'voc2coco', 'coco2voc', 'yolo2voc', 'voc2yolo',
           'bbox_iou',  'load_image'] #'draw_bboxes',

# Save to pickle
os.makedirs("D:/CustomData/result_file",exist_ok= True)
df.to_pickle("D:/CustomData/result_file/small_test_data.pkl")

# Read to pickle
# 만약 데이터가 완전히 생성되었다면 pickle file만 읽고 진행가능
df = pd.read_pickle("D:/CustomData/result_file/small_test_data.pkl")
print('read off')


# ************ Create Annotation ************

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

    # Create Annotation(YOLO)
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
        annots = np.concatenate([labels, bboxes_yolo], axis=1)
        string = annot2str(annots)
        f.write(string)

print('Missing:', miss_cnt)

train_files = train_path_list['img_path'].values
valid_files = valid_path_list['img_path'].values

# ************ Create text file for Yolov5 ************

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

# hyperparameter define
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

# python ./Yolov5/train.py --batch 8 --imgsz 1280 --epochs 10 --data 'D:/CustomData/file_path/data.yaml' --cfg 'yolov5s.yaml' --weights 'yolov5s.pt' --save-period 1 --patience 3 --project 'convenience' --label-smoothing 0.01 --optimizer 'AdamW' --hyp 'D:/CustomData/file_path/hym.yaml'
# python ./yolov5/detect.py --img 1280 --weights C:/Users/PC/Desktop/Yolo_con/convenience/exp48/weights/best.pt --source 'D:/CustomData/small_val' --save-txt --save-conf --conf 0.03 --iou-thres 0.4 --augment