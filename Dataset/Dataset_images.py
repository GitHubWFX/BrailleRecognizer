#生成数据图像
import os
import math
import random
import numpy as np
from PIL import Image
import cv2 as cv
from tqdm import trange	# 替换range()可实现动态进度条，可忽略
from matplotlib import pyplot as plt
from NeuralNetwork import tools
import pylab

PATH = "F:/BrailleFilePath/Dataset";

#生成空图像
def bulidModel(height, width):
    img = np.zeros((height, width), np.uint8);  #生成二值图像
    return img;

#把表示盲文的字符串变换为3*2的矩阵
def getBrailleArray(char_string):
    arr = np.zeros((3, 2), np.uint8);
    for i in range(3):   #
        arr[i][0] = int(char_string[i]);
    for i in range(3):   #
        arr[i][1] = int(char_string[i+3]);
    return  arr;

#绘制盲文
def drawBraille(char_string, height, width, size, height_b, width_b):
    # 对字符串反转，这样更便于我们对盲文系统的理解和使用
    # l = list(char_string);
    # l.reverse();
    # char_string = "".join(l);

    arr = getBrailleArray(char_string);  #生成矩阵
    #print(arr);
    img = bulidModel(height, width);    #生成背景
    h = round(height/3);
    w = round(width/2);
    # print("size:",height,width);
    #把图像分为3*2=6个方格，在其中心绘制盲符
    for i in range(3):
        for j in range(2):
            if arr[i][j]==1 :
                # 中心坐标
                y = round(h / 2 + i * height / 3 + height_b + (1 - i) * size / 3);
                x = round(w / 2 + j * width / 2 + width_b + (0.5 - j) * 2 * size / 3);  # 最后的数表示点距离的紧密度，数值越大越紧密
                # print(x, y);
                cv.circle(img, (x, y), size, 255, -1, 8);
    return img;

#添加噪声
def setNoisy(img, num, sert):
    (h,w) = img.shape;
    for i in range(num):
        x = random.randint(1, h-1);
        y = w-random.randint(1, w-1);
        img[x][y] = sert;

    return img;

def rotate(img, rotate):    #图像旋转
    img = Image.fromarray(img.astype('uint8'));
    img = img.rotate(rotate);
    img = np.asarray(img);
    return img

#随机的仿射变换形变
def deform(img):
    (h,w) = img.shape;
    for i in range(1):
        src = np.float32([[0, 0], [100, 0], [0, 100]]);
        dst = np.float32([[0+random.randint(-5,5), 0+random.randint(-5,5)], [100+random.randint(-5,5), 0+random.randint(-5,5)], [0+random.randint(-5,5), 100+random.randint(-5,5)]]);
        A1 = cv.getAffineTransform(src, dst)
        # 第三个参数：变换后的图像大小
        # 第四个参数：形变后的边界值，默认0
        img = cv.warpAffine(img, A1, (w, h), borderValue = 0)
    # 显示操作之后的图片
    return  img;

#构建盲文二值图像
def build(char_bin, height, width, size):
    img = drawBraille(char_bin, height, width, size,
                      random.randint(-round(height/25), round(height/25)),
                      random.randint(-round(width / 15), round(width / 15))
                      );
    img = rotate(img, random.randint(-2, 2));  # 随机旋转
    img = deform(img);
    # img = setNoisy(img,1000, 255);
    # img = setNoisy(img, 1000, 0);
    return img;

def main():
    height = 350;
    width = 250;
    size = 35;
    dataset_num = 1000;
    #64种点字
    print('Building images of braille characters');
    for c in trange(64):
        #创建目录
        trainPath = PATH + "/training-images/"+str(c);
        testPath = PATH+"/test-images/"+str(c);

        char_bin = tools.binString(c);
        if ~os.path.exists(trainPath):
            os.makedirs(trainPath);
        if ~os.path.exists(testPath):
            os.makedirs(testPath);

        #把标签保存到文件  a为追加写入  r为只读  w为覆盖写入
        f = open(PATH+"/batches.meta.txt", "a");
        f.write(char_bin+"\n");

        #每种点字生成n个图像,80%作为训练集，20%作为测试集
        s1 = round(dataset_num * 0.8);
        s2 = round(dataset_num * 0.2);
        for i in range(s1):
            img = build(char_bin, height, width, size+random.randint(-5,5));
            # tools.showImg("Image", img);
            img = Image.fromarray(img.astype('float32')); #转为Image类型
            img = img.resize((28, 28), Image.ANTIALIAS); #重整尺寸，不改变长宽比例
            img = np.asarray(img); #转为矩阵类型（个人比较喜欢矩阵操作）

            filename = trainPath + "/im" + str(i) + ".png";
            cv.imwrite(filename, img);  # 保存图像
        for i in range(s2):
            img = build(char_bin, height, width, size+random.randint(-10,10));
            # tools.showImg("Image", img);
            img = Image.fromarray(img.astype('float32')); #转为Image类型
            img = img.resize((28, 28), Image.ANTIALIAS); #重整尺寸，不改变长宽比例
            img = np.asarray(img); #转为矩阵类型（个人比较喜欢矩阵操作）

            filename = testPath + "/im" + str(s1+i) + ".png";
            cv.imwrite(filename, img);  # 保存图像

    print('\nAll finished!');


if __name__ == "__main__":
    #initdirs();

    main();


    # img = build(tools.binString(63), 350, 250, 40);
    # img = Image.fromarray(img.astype('float32'));  # 转为Image类型
    # img = img.resize((28, 28), Image.ANTIALIAS);  # 重整尺寸，不改变长宽比例
    # img = np.asarray(img);  # 转为矩阵类型（个人比较喜欢矩阵操作）
    # tools.showImg("Image", img);

