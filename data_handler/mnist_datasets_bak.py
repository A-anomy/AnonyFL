import torch
import torchvision
from torchvision import datasets,transforms
from torch.utils.data.dataset import Dataset
import gzip, os
import numpy as np
import struct
from matplotlib import pyplot as plt
from PIL import Image














# class MNIST(Dataset):
#     """
#     自定义FMNIST数据集读取，并使用DataLoader加载器加载数据
#     """
#     def __init__(self,root,train=True, transform=None, target_transform=None):
#         '''
#         data url : http://yann.lecun.com/exdb/mnist/

#         TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
#         [offset] [type] [value] [description]
#         0000 32 bit integer 0x00000803(2051) magic number #文件头魔数
#         0004 32 bit integer 60000 number of images #图像个数
#         0008 32 bit integer 28 number of rows #图像宽度
#         0012 32 bit integer 28 number of columns #图像高度
#         0016 unsigned byte ?? pixel #图像像素值
#         0017 unsigned byte ?? pixel
#         ……..
#         xxxx unsigned byte ?? pixel

#         TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
#         [offset] [type] [value] [description]
#         0000 32 bit integer 0x00000801(2049) magic number (MSB first)
#         0004 32 bit integer 60000 number of items
#         0008 unsigned byte ?? label
#         0009 unsigned byte ?? label
#         ……..
#         xxxx unsigned byte ?? label
#         The labels values are 0 to 9.
#         '''
#         super(FashionMNIST_IMG,self).__init__()
#         self.train = train
#         self.transform = transform
#         self.target_transform = target_transform

#         if self.train:    #train sets
#             images_file = root + r'\train-images.idx3-ubyte'
#             labels_file = root + r'\train-labels.idx1-ubyte'
#         else:
#             images_file = root + r'\t10k-images.idx3-ubyte'
#             labels_file = root + r'\t10k-labels.idx1-ubyte'


#         #读取二进制数据
#         offset1,offset2 = 0,0
#         fp_img = open(images_file,'rb').read()
#         fp_label = open(labels_file,'rb').read()

#         #解析文件头信息，依次为魔数、图片数量、每张图片的高、宽
#         magics1,num_img,rows,cols = struct.unpack_from('>IIII',fp_img,offset1)
#         magics2,num_label = struct.unpack_from('>II',fp_label,offset2)

#         #解析数据集
#         offset1 += struct.calcsize('>IIII')
#         offset2 += struct.calcsize('>II')
#         #img_fmt = '>'+str(rows*cols)+'B'    #图像数据像素值的类型为unsignedchar型，对应的format格式为B
#         #这里的图像大小为28*28=784，为了读取784个B格式数据，如果没有则只会读取一个值
#         #label_fmt = '>B'

#         self.images = np.empty((num_img,rows,cols))
#         self.labels = np.empty(num_label)

#         assert num_img==num_label   #判断图像个数是否等于标签个数，成立则往下执行

#         for i in range(num_img):
#             self.images[i] = np.array(struct.unpack_from('>'+str(rows*cols)+'B',fp_img,offset1)).reshape((rows,cols))
#             self.labels[i] = struct.unpack_from('>B',fp_label,offset2)[0]
#             offset1 +=struct.calcsize('>'+str(rows*cols)+'B')
#             offset2 += struct.calcsize('>B')

#     def __getitem__(self, item):
#         img = self.images[item]
#         label = self.labels[item]
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, label

#     def __len__(self):
#         return len(self.images)

#     def get_label(self,n):
#         """获得第n个数字对应的标签文本"""
#         text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#         return text_labels[int(n)]

#     def get_labels(self,labels):   #@save
#         """返回Fashion-MNIST数据集的所有标签文本
#         如labels = [1,3,5,3,6,2]
#         此函数具有迭代器功能
#         """
#         text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#         return [text_labels[int(i)] for i in labels]
