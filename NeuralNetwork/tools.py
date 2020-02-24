#本项目的工具集
import os
import cv2 as cv
import re

#把一个数字变为二进制字符串
def binString(n):
    char_bin = bin(n).replace("0b", "");  # 二进制数，表示盲点
    char_bin = char_bin.zfill(6);  # 左边补零
    return char_bin;

# 把字符串形式的数组转化为int数组
def Array_StringToInt(string):
    string = re.sub(r'[\[\]\s]*', '', string);  # 正则替换[、]、空格(\s)和换行(\n)  *表示替换全部
    string = string.split(',');
    array = [];
    for char in string:
        array.append(int(char));
    return array;

# 读取文件夹下的所有文件
def read_directory(directory_name):
    array_of_file = [];
    for filename in os.listdir(directory_name):
        array_of_file.append(directory_name + "/" + filename);
    return array_of_file;


#在窗口中打开图片
def showImg(title, img):
    cv.namedWindow(title)
    cv.imshow(title, img)
    cv.waitKey(0);




