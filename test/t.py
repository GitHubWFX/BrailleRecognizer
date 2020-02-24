
INPUT = 28*28;  #输入层维度
OUTPUT = 64;    #输出层维度
HIDDEN = [1568,1568];   #隐藏层维度
arr = [INPUT] + HIDDEN[:] +[OUTPUT];
print(arr);