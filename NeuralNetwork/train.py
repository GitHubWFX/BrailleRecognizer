#训练
from BPNN import NeuralNetwork
from MnistReader import  MnistReader
from tqdm import trange
import numpy as np
import pickle
import csv

BasePath = 'F:/BrailleFilePath/Dataset'; #文件目录路径
INPUT = 28*28;  #输入层维度
OUTPUT = 64;    #输出层维度
HIDDEN = [784,784];   #隐含层维度

def train():
    # 数据集为64*1000张带标签的28x28盲文符号图像
    mr = MnistReader(BasePath);
    x_len, x, y = mr.load_dataset('train-images.idx3-ubyte','train-labels.idx1-ubyte');
    # x_len = 100;
    # x = x[:x_len];
    # y = y[:x_len];
    print('训练集：%d' % (x_len));
    #numpy格式矩阵
    x = np.array(x);
    y = np.array(y);

    nn = NeuralNetwork( [INPUT] + HIDDEN[:] +[OUTPUT] )	# 神经网络各层神经元个数
    #训练
    nn.fit(x, y);
    print('训练结束');

    # self = nn.getNeuralNetwork();
    # print('layers:',self.layers);
    # print('fittimes:', self.fittimes);
    # print('weights:',self.weights);
    # print('bias:',self.bias);

    # 保存模型
    file = open(BasePath+'/NN_Model.txt', 'wb')
    pickle.dump(nn, file)

def predict():
    mr = MnistReader(BasePath);
    x_t_len, x_t, y_t = mr.load_dataset('test-images.idx3-ubyte', 'test-labels.idx1-ubyte');
    # 读取模型
    file = open(BasePath + '/NN_Model.txt', 'rb');
    nn = pickle.load(file);

    # self = nn.getNeuralNetwork();
    # print('layers:',self.layers);
    # print('fittimes:', self.fittimes);
    # print('weights:',self.weights);
    # print('bias:',self.bias);

    count = 0;
    print('测试集：%d' %x_t_len);
    for i in range(x_t_len):
        p, _ = nn.predict(x_t[i]);
        if p == y_t[i]:
            count += 1;
        if((i+1)%500==0):
                print('已进行%d个次预测，正确率为%.3f%%'%(i+1,(count+0.000001)/i*100));

    print('模型识别正确率：%.3f%%'%(count/x_t_len*100));


#train();
predict();
