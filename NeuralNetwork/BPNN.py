import numpy as np
from tqdm import trange

# 激活函数采用Sigmoid
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX));

# Sigmoid的导数
def sigmoid_derivative(inx):
    return sigmoid(inx) * (1 - sigmoid(inx));

class NeuralNetwork:	# 神经网络
    def __init__(self, layers):	# layers为神经元个数列表
        '''
        :param layers: 
            layers为表示神经网络神经元的矩阵
            矩阵第一个元素表示输入层节点数目
            最后一个元素表示输出层节点数目
            其余为隐含层节点数目
            
            设以手写数字数据集mnist，layers=[784,1568,1568,10]为例子
            784是输入层节点数量，等同数据大小，手写数字的数据维度为28*28=784
            10是输出层节点数量，等同分类数，手写数字0~9，为10个类别
            两个1568是隐含层节点数量，BP神经网络中，隐含层节点数目是没有固定的，可以随意设置
            在确定隐层节点数时必须满足：隐层节点数必须小于N-1，训练样本数必须多于网络模型的连接权数，一般为2~10倍
        '''
        self.layers = layers;
        self.activation = sigmoid; # 激活函数
        self.activation_deriv = sigmoid_derivative; # 激活函数导数
        self.weights = []; # 权重列表
        self.bias = []; # 偏置列表
        self.fittimes = 0;  #记录训练次数
        for i in range(1, len(layers)):	# 正态分布初始化
            #初始化权重和偏置，初始权重必须不能全为0
            self.weights.append(np.random.randn(layers[i-1], layers[i]))
            self.bias.append(np.random.randn(layers[i]))

    #self是python类函数的参数，保存类的内存地址，可以用来保存成员变量和函数
    #调用函数时，不需要填写self参数
    # self的名称可以任意修改，但一般都是有“self”，这是一种习惯
    #self类似c++的指针，返回self可以获取权重偏置等数据
    def getNeuralNetwork(self):
        return self;

    #训练
    def fit(self, x, y, epochs=1, learning_rate=0.2):	# 反向传播算法
        x = np.atleast_2d(x); #维度改变   atleast_xd 支持将输入数据直接视为 x维。
        n = len(y);	# 样本数
        y = np.array(y);

        for p in range(epochs): #样本过少时根据epochs减半学习率,epochs默认为1，即循环1次
            for k in trange(n):
                self.fittimes+=1;
                if (k+1) % n == 0:
                    learning_rate *= 0.5;	# 每训练完一代,样本减半学习率

                a = [ x[k%n] ];	# 保存各层激活值的列表

                # 正向传播开始  #np.dot(a,b)是矩阵乘法运算的函数  a*b则是矩阵的点乘运算
                for lay in range(len(self.weights)):
                    #numpy dot()函数是两个数组的点乘运算
                    a.append(self.activation(np.dot(a[lay], self.weights[lay]) + self.bias[lay]));

                # 反向传播开始
                label = np.zeros(a[-1].shape);
                label[ y[k%n] ] = 1;    # 根据类号生成标签

                #损失函数
                error = label - a[-1];	# 误差值
                loss = [ error*self.activation_deriv(a[-1]) ];	# 保存各层误差值的列表

                layer_num = len(a) - 2;	# 导数第二层开始
                for j in range(layer_num, 0, -1):
                    loss.append(loss[-1].dot(self.weights[j].T) * self.activation_deriv(a[j]));	# 误差的反向传播
                loss.reverse();#数组倒序

                for i in range(len(self.weights)):	# 正向更新权值
                    layer = np.atleast_2d(a[i]);
                    delta = np.atleast_2d(loss[i]);
                    self.weights[i] += learning_rate * layer.T.dot(delta);
                    self.bias[i] += learning_rate * loss[i];

    #预测
    def predict(self, x):
        a = np.array(x, dtype=np.float);
        for lay in range(len(self.weights)):	# 正向传播
            a = self.activation(np.dot(a, self.weights[lay]) + self.bias[lay]);
        a = list(100 * a/sum(a));	# 改为百分比显示
        i = a.index(max(a));	# 预测值
        per = [];	# 各类的置信程度
        for num in a:
            per.append(str(np.round(num, 2))+'%');
        return i, per;

