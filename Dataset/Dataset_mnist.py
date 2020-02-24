#生成训练集和测试集
# 把png图像生成mnist数据集
# 图像分辨率为28*28
# 训练集图像保存在training-images
# 测试集图像保存在test-images
# 标签名保存在batches.meta.txt
import os
from PIL import Image
from array import *
from random import shuffle
from tqdm import trange

# Load from and save to
Base_PATH = 'F:/BrailleFilePath/Dataset';   #图像地址
# Save_PATH = '../MNIST_data_braille'; #数据集保存路径
Save_PATH = 'F:/BrailleFilePath/Dataset';

Names = [
    [Base_PATH + "/training-images", "train"],
    [Base_PATH + "/test-images", "test"]
];

print('build mnist started:');

for name in Names:
    print('building %s ' % name[1]);

    data_image = array('B');
    data_label = array('B');
    FileList = [];
    for dirname in os.listdir(name[0])[0:]:  # 从索引0开始获取所有目录
        path = os.path.join(name[0], dirname)  # 保存目录名(路径)
        for filename in os.listdir(path):  # 获取png文件的文件名
            if filename.endswith(".png"):
                FileList.append(os.path.join(name[0], dirname, filename));

    shuffle(FileList);  # 打乱顺序

    #FileList = FileList[0:64 * 1000];   #只取64*1000个图像制作数据集

    for i in trange(len(FileList)):
        filename = FileList[i];
        #print(filename)
        label = int(filename.split('\\')[1]);

        Im = Image.open(filename);

        pixel = Im.load();

        width, height = Im.size;

        for x in range(0, width):
            for y in range(0, height):
                data_image.append(pixel[y, x]);

        data_label.append(label);  # labels start (one unsigned byte each)

    hexval = "{0:#0{1}x}".format(len(FileList), 6);  # number of files in HEX

    # header for label array

    header = array('B');
    header.extend([0, 0, 8, 1, 0, 0]);
    header.append(int('0x' + hexval[2:][:2], 16));
    header.append(int('0x' + hexval[2:][2:], 16));

    data_label = header + data_label;

    # additional header for images array

    if max([width, height]) <= 256:
        header.extend([0, 0, 0, width, 0, 0, 0, height])
    else:
        raise ValueError('Image exceeds maximum size: 256x256 pixels');

    header[3] = 3  # Changing MSB for image data (0x00000803)

    data_image = header + data_image

    output_file = open(Save_PATH + '/' + name[1] + '-images.idx3-ubyte', 'wb')
    data_image.tofile(output_file)
    output_file.close()
    print('saved %s' % (Save_PATH + '/' + name[1] + '-images.idx3-ubyte'));

    output_file = open(Save_PATH + '/' + name[1] + '-labels.idx1-ubyte', 'wb')
    data_label.tofile(output_file)
    output_file.close()
    print('saved %s' % (Save_PATH + '/' + name[1] + '-labels.idx1-ubyte'));

    print('finished')

print('All finished!')

# 用gzip压缩文件
# for name in Names:
# 	os.system('gzip '+name[1]+'-images.idx3-ubyte')
# 	os.system('gzip '+name[1]+'-labels.idx1-ubyte')
