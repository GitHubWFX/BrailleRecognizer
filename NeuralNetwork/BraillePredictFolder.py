#识别目录下的所有图像
# encoding: utf-8
import sys
import pickle
import numpy as np
import json
import cv2 as cv
from PIL import Image
from tqdm import trange
from MnistReader import MnistReader
from BPNN import NeuralNetwork
import tools as tl
import Predict as pd

if __name__ == '__main__':
    foldername = sys.argv[1];
    #foldername = 'F:/BrailleFilePath/sys_cache/paragraph/5_1';
    files = tl.read_directory(foldername);
    images_array = [];
    for fn in files:
        if fn.endswith(".png") | fn.endswith(".jpg"):
            img = Image.open(fn);
            img = np.array(img);
            img = pd.adapt(img);
            array = img.reshape(28*28);
            # pre = pd.predict(array);
            # print(pre)
            # tl.showImg(fn,img);
            images_array.append(array);
    res = pd.predictArrays(images_array);
    for i in range(len(res)):
        print(res[i])

    # json_ = {
    #     "predict": {
    #         "result" : res
    #     }
    # }
    # print(json_);

