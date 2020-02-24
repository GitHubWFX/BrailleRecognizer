#解析，查看Mnist格式文件内容
import numpy as np
import struct
import cv2 as cv

class MnistReader:
    def __init__(self,path):
        self.BasePath = path+'/'

    def read_labels(enter):
        data = enter.read()
        head = struct.unpack_from('>II', data, 0)
        number = head[1]
        offset = struct.calcsize('>II')
        numString = '>' + str(number) + "B"
        labels = struct.unpack_from(numString, data, offset)
        labels = np.reshape(labels, [number])
        return number,labels

    def read_images(enter):
        data = enter.read()
        head = struct.unpack_from('>IIII', data, 0)
        offset = struct.calcsize('>IIII')
        number = head[1]
        width = head[2]
        height = head[3]
        bits = number * width * height
        bitsString = '>' + str(bits) + 'B'
        images = struct.unpack_from(bitsString, data, offset)
        images = np.reshape(images, [number, width *height])
        return number,images    #图像数量、图像矩阵(img.shape=[1,w*h])的集合，需要重整为方形

    #指定数量的读取(size<=0表示获取全部)
    def load_labers(self,filename):
        enter = open(self.BasePath+filename,'rb');
        num,labels = MnistReader.read_labels(enter);
        # print(num,labels);
        return num, labels;

    def load_images(self,filename):
        enter = open(self.BasePath+filename,'rb')
        num, images = MnistReader.read_images(enter)
        # print(num,images.shape);
        return num, images;

    def load_dataset(self,images_filename, label_filename):
        num, images = MnistReader.load_images(self,images_filename);
        num, label = MnistReader.load_labers(self,label_filename);
        return num, images, label;


# mr = MnistReader('F:/BrailleFilePath/Dataset');
# trainN, trainL = mr.load_labers('train-labels.idx1-ubyte');
# trainN, trainI = mr.load_images('train-images.idx3-ubyte');
# testN, testL = mr.load_labers('test-labels.idx1-ubyte');
# testN, testI = mr.load_images('test-images.idx3-ubyte');
#num, trainI, trainL = mr.load_dataset('train-images.idx3-ubyte','train-labels.idx1-ubyte')
# print(trainN)
# print(len(trainI))
# print(trainI.shape)

# index = 10;
# label = trainL[index];
# img = trainI[index];
# img = img.astype('uint8').reshape(28,28);  # 转为Image类型
# title = bin(label).replace("0b", "").zfill(6);
# print(label, title, '\n');
# cv.imshow(title, img)
# cv.waitKey(0);







