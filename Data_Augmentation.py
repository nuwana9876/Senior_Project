import albumentations as A
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import shutil
import os

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
], bbox_params= A.BboxParams(format = 'pascal_voc', label_fields=['category_ids']))

def Augmentation_(path,bbox,bbox_list,img_path,label_list,width_list,height_list,name,path_list,idx):
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

    transformed = transform(image=img_copy, bboxes=bbox, category_ids=category_ids , cropping_bbox = bbox[0])
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
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
    # 저장 하는 코드
        extension = os.path.splitext(transformed_img_path)[1]  # 이미지 확장자
        result, encoded_img = cv2.imencode(extension, transformed_image)
        if result:
            with open(transformed_img_path, mode='w+b') as f:
                encoded_img.tofile(f)

        img_path.append(transformed_img_path)
        bbox_list.append(transformed_bboxes)
        label_list.append(label)
        width_list.append(img.shape[1])  # img.shape -> h,w,c로 구성
        height_list.append(img.shape[0])
        name.append(pre_trans_name)
        path_list.append(destination)

