# encoding: utf-8
import pickle
import math
import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import trange
import tools as tl

BasePath = 'F:/BrailleFilePath/Dataset';

def predict(img):  # 读取测试集，预测，画图
    # 读取模型
    file = open(BasePath + '/NN_Model.txt', 'rb')
    nn = pickle.load(file)
    # 预测
    pre, lst = nn.predict(img);
    pre = tl.binString(pre);
    return pre;


def predictArrays(array):  # 读取测试集，预测，画图
    '''
    :param array: 28*28图像的数组
    :return:
    '''
    res = [];
    # 读取模型
    file = open(BasePath + '/NN_Model.txt', 'rb')
    nn = pickle.load(file)
    # 预测
    for img in array:
        pre, lst = nn.predict(img);
        pre = tl.binString(pre);
        res.append(pre);
        # print(img,pre);
    return res;

# 适配图像  输入一个nunpy矩阵，用0填充边缘使其常宽比例为1:1，然后重整为[28,28]的矩阵
def adapt(img):
    '''
    :param img: 图像矩阵
    :return: 图像矩阵
    '''
    a, b = img.shape;  # a高  b宽
    mx = max(a, b);
    mn = min(a, b);
    d = math.ceil((mx - mn) / 2);
    if a < b:  # 需要增加高度
        imgnp = cv.copyMakeBorder(img, d, d - 1, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])
    else:  # 需要增加宽度
        imgnp = cv.copyMakeBorder(img, 0, 0, d, d - 1, cv.BORDER_CONSTANT, value=[0, 0, 0])
    img = Image.fromarray(img.astype('uint8'));
    img = img.resize([28, 28]);
    img = np.array(img);

    return img;