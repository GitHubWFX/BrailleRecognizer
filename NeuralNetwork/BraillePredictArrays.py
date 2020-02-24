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
    # images_array = sys.argv[1];
    images_array = [];
    res = pd.predictArrays(images_array);
    for i in range(len(res)):
        print(res[i])

    # json_ = {
    #     "predict": {
    #         "result" : {}
    #     }
    # }
    # for i in range(len(res)):
    #     json_["predict"]["result"][str(i)] = res[i];
    # print(json_);

