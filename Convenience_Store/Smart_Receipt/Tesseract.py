# Reference
# 1. https://github.com/jin05154/edge-detection/blob/main/edge-detection-perspective-transform.py

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
import numpy as np
import random
import wandb
import torch
import random
import cv2
from cv2 import dnn_superres
from PIL import Image
from pytesseract import *

def image_threshold(image):
    result = cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return result

def remove_noise(image,kernel_size = 5):
    result = cv2.medianBlur(image,ksize = kernel_size)
    return result

def Tesseract_OCR():

    config = ('--oem 3 --psm 6')

    pytesseract.tesseract_cmd = R'C:/Program Files/Tesseract-OCR/tesseract'
    filename = './result.jpg'
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    # kernel = np.ones((1, 1), np.uint8)
    # image = cv2.erode(image,kernel,iterations = 1)
    # kernel = np.ones((2,2),np.uint8)
    # image = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
    text = image_to_string(image,lang = 'kor+eng', config= config)
    print(text)

    print('이미지 gray 처리')
    img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # kernel = np.ones((1, 1), np.uint8)
    # img_gray = cv2.erode(img_gray,kernel,iterations = 1)
    # kernel = np.ones((2,2),np.uint8)
    # img_gray = cv2.morphologyEx(img_gray,cv2.MORPH_OPEN,kernel)
    cv2.imshow('grayscale', img_gray)
    cv2.waitKey(0)
    print(image_to_string(img_gray, lang='kor+eng', config=config))

    print('image_threshold')
    img_gray = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((1, 1), np.uint8)
    img_gray = cv2.erode(img_gray,kernel,iterations = 2)
    _, result = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    kernel = np.ones((1,1),np.uint8)
    result = cv2.morphologyEx(result,cv2.MORPH_OPEN,kernel)
    # result = cv2.medianBlur(result, 3)
    cv2.imshow('grayscale',result)
    cv2.waitKey(0)
    print(image_to_string(result,lang = 'kor+eng',config = config))

#에지 검출 : 흑백 -> 가우시안블러링 -> 캐니
def edge_detection(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.edgePreservingFilter(gray, flags=1, sigma_s=45, sigma_r=0.2)

    edged = cv2.Canny(gray, 75, 200, True)
    # cv2.imshow("grayscale", gray)
    cv2.waitKey(0)
    cv2.imshow("edged", edged)
    cv2.waitKey(0)

    return edged

filename = "C:/Users/PC/Desktop/down/down11.jpg"

def contoursGrab(edged):
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]  # contourArea : contour가 그린 면적

    largest = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    r = cv2.boxPoints(rect)
    box = np.int0(r)

    size = len(box)

    # 2.A 원래 영상에 추출한 4 변을 각각 다른 색 선분으로 표시한다.
    cv2.line(resize_img, tuple(box[0]), tuple(box[size - 1]), (255, 0, 0), 3)
    for j in range(size - 1):
        color = list(np.random.random(size=3) * 255)
        cv2.line(resize_img, tuple(box[j]), tuple(box[j + 1]), color, 3)

    # 4개의 점 다른색으로 표시
    boxes = [tuple(i) for i in box]
    cv2.line(resize_img, boxes[0], boxes[0], (0, 0, 0), 15)  # 검
    cv2.line(resize_img, boxes[1], boxes[1], (255, 0, 0), 15)  # 파
    cv2.line(resize_img, boxes[2], boxes[2], (0, 255, 0), 15)  # 녹
    cv2.line(resize_img, boxes[3], boxes[3], (0, 0, 255), 15)  # 적
    print(resize_img)
    cv2.imshow("With_Color_Image", resize_img)

    return boxes

def grab_cut(resized):
    mask_img = np.zeros(resized.shape[:2], np.uint8)  # 초기 마스크를 만든다.

    # grabcut에 사용할 임시 배열을 만든다.
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # rect = (130, 51, 885-130, 661-51) #mouse_handler로 알아낸 좌표 / card1일때
    rect = (107, 89, 580, 467)  # card2 일 때
    rect = (100, 100, 400, 400)
    cv2.grabCut(resized, mask_img, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)  # grabcut 실행
    mask_img = np.where((mask_img == 2) | (mask_img == 0), 0, 1).astype('uint8')  # 배경인 곳은 0, 그 외에는 1로 설정한 마스크를 만든다.
    img = resized * mask_img[:, :, np.newaxis]  # 이미지에 새로운 마스크를 곱해 배경을 제외한다.

    background = resized - img

    background[np.where((background >= [0, 0, 0]).all(axis=2))] = [0, 0, 0]

    img_grabcut = background + img

    cv2.imshow('grabcut', img_grabcut)
    cv2.waitKey(0)

    new_edged = edge_detection(img_grabcut)
    cv2.waitKey(0)

    global new_contour
    new_contour = contoursGrab(new_edged)

src = []  # 명함 영역 꼭지점의 좌표

def solving_vertex(pts):
    points = np.zeros((4,2), dtype= "uint32") #x,y쌍이 4개 쌍이기 때문

    #원점 (0,0)은 맨 왼쪽 상단에 있으므로, x+y의 값이 제일 작으면 좌상의 꼭짓점 / x+y의 값이 제일 크면 우하의 꼭짓점
    s = pts.sum(axis = 1)
    points[0] = pts[np.argmin(s)] #좌상
    points[3] = pts[np.argmax(s)] #우하

    #원점 (0,0)은 맨 왼쪽 상단에 있으므로, y-x의 값이 가장 작으면 우상의 꼭짓점 / y-x의 값이 가장 크면 좌하의 꼭짓점
    diff = np.diff(pts, axis = 1)
    points[2] = pts[np.argmin(diff)] #우상
    points[1] = pts[np.argmax(diff)] #좌하

    src.append(points[0])
    src.append(points[1])
    src.append(points[2])
    src.append(points[3])

    return points

def transformation():
    #print(src)
    src_np = np.array(src, dtype=np.float32)
    #print(src_np)
    dst_np = np.array([
    [0, 0],
    [0, 1280],
    [400, 0],
    [400, 1280]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)
    result = cv2.warpPerspective(resize_img, M=M, dsize=(400, 1280))
    cv2.imshow("result", result)

    # # 모델 로드하기
    # sr = dnn_superres.DnnSuperResImpl_create()
    # sr.readModel('./EDSR_x3.pb')
    # sr.setModel('edsr', 3)
    # # 이미지 추론하기 ( 해당 함수는 전처리와 후처리를 함꺼번에 해준다)
    # result = sr.upsample(result)

    cv2.imwrite("result.jpg",result)
    cv2.waitKey(0)


def transformationGrab(resized, pts):
    p = np.array(pts)
    rect = np.zeros((4, 2), dtype="float32")
    s = p.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    a, b, c, d = rect
    w1 = abs(c[0] - d[0])
    w2 = abs(a[0] - b[0])
    h1 = abs(b[1] - c[1])
    h2 = abs(a[1] - d[1])
    w = max([w1, w2])
    h = max([h1, h2])
    #dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.array([
    [0, 0],
    [400, 0],
    [400, 1280],
    [0, 1280]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    result = cv2.warpPerspective(resized, M, dsize=(400, 1280))
    cv2.imshow("transformation", result)

    # # 모델 로드하기
    # sr = cv2.dnn_superres.DnnSuperResImpl_create()
    # sr.readModel('./EDSR_x3.pb')
    # sr.setModel('edsr', 3)
    # # 이미지 추론하기 ( 해당 함수는 전처리와 후처리를 함꺼번에 해준다)
    # result = sr.upsample(result)

    cv2.imwrite("result.jpg", result)
    cv2.waitKey(0)
    return result

def contours(edge):
    global checkpnt
    #edged = edge_detection(resize_img)
    (cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 계층관계가 필요없기 때문에 contour만 추출

    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]  # contourArea : contour가 그린 면적

    for i in cnts:
        peri = cv2.arcLength(i, True)  # contour가 그리는 길이 반환
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # 길이에 2% 정도 오차를 둔다

        if len(approx) == 4:  # 도형을 근사해서 외곽의 꼭짓점이 4개라면 명암의 외곽으로 설정
            screenCnt = approx
            size = len(screenCnt)
            break
        if len(approx) != 4 and checkpnt == 0:  # 사각형이 그려지지 않는다면 grab_cut 실행
            size = 0
            checkpnt += 1
            grab_cut(resize_img)

        if len(approx) != 4 and checkpnt > 0:
            size = 0

    if (size > 0):
        # 2.A 원래 영상에 추출한 4 변을 각각 다른 색 선분으로 표시한다.
        cv2.line(resize_img, tuple(screenCnt[0][0]), tuple(screenCnt[size-1][0]), (255, 0, 0), 3)
        for j in range(size-1):
            color = list(np.random.random(size=3) * 255)
            cv2.line(resize_img, tuple(screenCnt[j][0]), tuple(screenCnt[j+1][0]), color, 3)

        #for i in screenCnt: #이렇게 하면 네 변을 다른 색으로 표현 불가능(네 변이 모두 다 똑같은 색으로 나온다.)
            #color = list(np.random.random(size=3) * 256)
            #cv2.drawContours(resize_img, [screenCnt], -1,color, 3)

        # 2.B 추출된 선분(좌, 우, 상, 하)의 기울기, y절편, 양끝점의 좌표를 각각 출력
        axis = np.zeros(4)

        # 기울기 = (y증가량) / (x증가량)
        #left_axis = (screenCnt[0][0][1] - screenCnt[1][0][1]) / (screenCnt[0][0][0] - screenCnt[1][0][0])
        #down_axis = (screenCnt[1][0][1] - screenCnt[2][0][1]) / (screenCnt[1][0][0] - screenCnt[2][0][0])
        #right_axis = (screenCnt[2][0][1] - screenCnt[3][0][1]) / (screenCnt[2][0][0] - screenCnt[3][0][0])
        #upper_axis = (screenCnt[3][0][1] - screenCnt[0][0][1]) / (screenCnt[3][0][0] - screenCnt[0][0][0])

        axis[3] = (screenCnt[3][0][1] - screenCnt[0][0][1]) / (screenCnt[3][0][0] - screenCnt[0][0][0])
        for k in range(3):
            axis[k] = (screenCnt[k][0][1] - screenCnt[k+1][0][1]) / (screenCnt[k][0][0] - screenCnt[k+1][0][0])

        left_axis = axis[0] #좌 기울기
        down_axis = axis[1] #하 기울기
        right_axis = axis[2] #우 기울기
        upper_axis = axis[3] #상 기울기

        print("(2.B) 순서대로 좌, 우, 상, 하 선분의 기울기")
        print(left_axis, right_axis, upper_axis, down_axis)
        print("\n")

        # y = ax + b 에서 x = 0일때의 b가 y절편 / 기울기를 알고 두 좌표를 알 때의 방정식 : y - y1 = (y2 - y1)/(x2 - x1) * (x - x1)
        #좌 선분의 y절편
        #left_y - screenCnt[1][0][1] = left_axis * (left_x - screenCnt[1][0][0])
        #left_y = (left_axis * left_x) - (left_axis * screenCnt[1][0][0]) + screenCnt[1][0][1]
        #따라서 left_y = screenCnt[1][0][1] - (left_axis * screenCnt[1][0][0])
        left_y = screenCnt[1][0][1] - (left_axis * screenCnt[1][0][0]) #좌 y절편

        #우 선분의 y절편
        #right_y - screenCnt[3][0][1] = right_axis * (right_x - screenCnt[3][0][0])
        #right_y = (right_axis * right_x) - (right_axis * screenCnt[3][0][0]) + screenCnt[3][0][1]
        #따라서 right_y = screenCnt[3][0][1] - (right_axis * screenCnt[3][0][0])
        right_y = screenCnt[3][0][1] - (right_axis * screenCnt[3][0][0]) #우 y절편

        #상 선분의 y절편
        #upper_y - screenCnt[0][0][1] = upper_axis * (upper_x - screenCnt[0][0][0])
        #upper_y = (upper_axis * upper_x) - (upper_axis * screenCnt[0][0][0]) + screenCnt[0][0][1]
        #따라서 upper_y = screenCnt[0][0][1] - (upper_axis * screenCnt[0][0][0])
        upper_y = screenCnt[0][0][1] - (upper_axis * screenCnt[0][0][0]) #상 y절편

        #하 선분의 y절편
        #donw_y - screenCnt[2][0][1] = down_axis * (down_x - screenCnt[2][0][0])
        #down_y = (down_axis * down_x) - (down_axis * screenCnt[2][0][0]) + screenCnt[2][0][1]
        #따라서 down_y = screenCnt[2][0][1] - (down_axis * screenCnt[2][0][0])
        down_y = screenCnt[2][0][1] - (down_axis * screenCnt[2][0][0]) #하 y절편

        print("(2.B) 순서대로 좌, 우, 상, 하 선분의 y절편")
        print(left_y, right_y, upper_y, down_y)
        print("\n")

        #양끝점의 좌표
        print("(2.B) 순서대로 좌, 우, 상, 하 선분의 양 끝점")
        print((screenCnt[0][0][0], screenCnt[0][0][1]), (screenCnt[1][0][0], screenCnt[1][0][1])) #좌 선분의 양 끝점
        print((screenCnt[2][0][0], screenCnt[2][0][1]), (screenCnt[3][0][0], screenCnt[3][0][1])) #우 성분의 양 끝점
        print((screenCnt[0][0][0], screenCnt[0][0][1]), (screenCnt[3][0][0], screenCnt[3][0][1])) #상 성분의 양 끝점
        print((screenCnt[1][0][0], screenCnt[1][0][1]), (screenCnt[2][0][0], screenCnt[2][0][1])) #하 성분의 양 끝점
        print("\n")

        # 3.B 네 꼭짓점을 각각 다른 색 점으로 표시한다.
        cv2.drawContours(resize_img, screenCnt, 0, (0, 0, 0), 15) #검
        cv2.drawContours(resize_img, screenCnt, 1, (255, 0, 0), 15) #파
        cv2.drawContours(resize_img, screenCnt, 2, (0, 255, 0), 15) #녹
        cv2.drawContours(resize_img, screenCnt, 3, (0, 0, 255), 15) #적

        cv2.imshow("With_Color_Image", resize_img)
        x = np.array(resize_img)
        img2 = Image.fromarray(x)
        img2.save('resize_img.jpg','JPEG')
        print('resize_img :',resize_img)
        print('-------------------------------------------------------')

        # 3.C  네 꼭지점(좌상, 좌하, 우상, 우하)의 좌표를 출력한다.
        vertex = solving_vertex(screenCnt.reshape(4,2))
        #(topLeft, bottomLeft, topRight, bottomRight) = vertex

        print("(3.C) 순서대로 좌상, 좌하, 우상, 우하의 꼭짓점 좌표")
        print(vertex)


global checkpnt
checkpnt = 0
global new_contour
img = cv2.imread(filename,cv2.IMREAD_COLOR)
resize_img = cv2.resize(img, dsize=(960, 1280), interpolation=cv2.INTER_AREA)
edged = edge_detection(resize_img)
contours(edged)
if checkpnt == 0:
    transformation()
else:
    pts = new_contour
    transformationGrab(resize_img,pts)

Tesseract_OCR()