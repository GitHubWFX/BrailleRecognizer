#神经网络测试
import pickle
import numpy as np
from PIL import Image
from tqdm import trange
from matplotlib import pyplot as plt
from MnistReader import MnistReader
from BPNN import NeuralNetwork
import tools

BasePath = 'F:/BrailleFilePath/Dataset';

def diplay_test():	# 读取测试集，预测，画图
    #读取测试集
    mr = MnistReader(BasePath)
    _, x_t = mr.load_images('test-images.idx3-ubyte')

    x_t = x_t[:50];
    print(x_t[0])
    #读取模型
    file = open(BasePath+'/NN_Model.txt', 'rb')
    nn = pickle.load(file)
    print('开始测试')
    for i in trange(len(x_t)):
        img = x_t[i];
        # 预测
        pre, lst = nn.predict(img);
        pre = tools.binString(pre);
        #打开图像
        img = np.array(img, dtype=np.uint8);
        img = img.reshape(28, 28);
        plt.imshow(img, cmap='gray');
        plt.title(pre, fontsize=24);
        plt.axis('off');

        #保存图像
        plt.savefig(BasePath+'/predict/img_' + str(i) + '_' + pre + '.png');

    print('测试结束')

if __name__ == '__main__':
    diplay_test();




