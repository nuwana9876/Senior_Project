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

# 구현 해야할 part 분류
# 1. 영수증 사진이 입력으로 들어오면 이를 opencv로 영수증의 글자를 text로 변환해주는 코드를 작성
# 2. database로부터 처음에 입력된 상품의 이름과 영수증의 text와 비교해서 사려고 한 물품과 개수가 맞는지를 확인해주는 코드 작성
# 3. 만약 개수가 다르거나, 구매한 물품의 종류가 다른 경우, 이를 확인해 사용자에게 계산이 잘못되었다는 것과 어느 상품이 잘못 계산되었는지 확인해주는 코드 작성
# 4. 음성 API 사용하는 코드 작성
# 5. MYSQL과 연동할 수 있는 코드 작성 (음성입력 저장 및 불러오기 task) # https://lucathree.github.io/python/day16/
















