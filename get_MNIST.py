# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 19:27:49 2017

@author: xiao
"""

"""
读入训练，测试的手写数字数据
"""

import struct
import numpy as np
import os

os.chdir(r"C:\Users\xiao\.spyder-py3\机器学习")

# 解析函数，将mnist数据集解析为我们需要的数据
# eg: filrname = 'train-images-idx3-ubyte'
#     train_X = Analytic_fun(filename)
    
# 下面两个函数是我在网上找到的，http://blog.csdn.net/qq_32166627/article/details/62218072    
def loadImageSet(filename):  
  
    binfile = open(filename, 'rb') # 读取二进制文件  
    buffers = binfile.read()  
  
    head = struct.unpack_from('>IIII', buffers, 0) # 取前4个整数，返回一个元组,I表示int数据  
  
    offset = struct.calcsize('>IIII')  # 定位到data开始的位置  
    imgNum = head[1]  
    width = head[2]  
    height = head[3]  
  
    bits = imgNum * width * height  # data一共有60000*28*28个像素值  
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'  
  
    imgs = struct.unpack_from(bitsString, buffers, offset) # 取data数据，返回一个元组  
  
    binfile.close()  
    imgs = np.reshape(imgs, [imgNum, width * height]) # reshape为[60000,784]型数组  
  
    return imgs  
  
  
def loadLabelSet(filename):  
  
    binfile = open(filename, 'rb') # 读二进制文件  
    buffers = binfile.read()  
  
    head = struct.unpack_from('>II', buffers, 0) # 取label文件前2个整形数  
  
    labelNum = head[1]  
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置  
  
    numString = '>' + str(labelNum) + "B" # fmt格式：'>60000B'  
    labels = struct.unpack_from(numString, buffers, offset) # 取label数据  
  
    binfile.close()  
    labels = np.reshape(labels, [labelNum]) # 转型为列表(一维数组)  
  
    return labels  


def train_lsc(train_x,train_y):
    m,n = train_x.shape
    X = np.zeros((m,n+1))
    X[:,0:n] = train_x
    X[:,n] = np.ones((1,m))
    train_X = X
    train_Y = train_y
    W = np.dot(np.dot(np.linalg.pinv(np.dot(train_X.T,train_X)),train_X.T),train_Y)
    return W
    

def OFK(train_y,K):
    (m,) = train_y.shape
    train_Y = np.zeros((m,len(K)))
    for j in range(0,m):
        for i in range(0,len(K)):
            if train_y[j] == K[i]:
                train_Y[j,i] = 1
    return train_Y

def test_lsc(test_x,W):
    m,n = test_x.shape
    X = np.zeros((m,n+1))
    X[:,0:n] = test_x
    X[:,n] = np.ones((1,m))
    test_X = X
    pre_Y = np.dot(test_X,W)
    return pre_Y

    
def one_of_kind(y,K):
    m,n = y.shape
    Y = np.zeros((m,len(K)))
    for i in range(0,m):
        j = np.argmax(y[i,:])
        Y[i,j] = 1
    return Y

    
    
# Train_x->60000*784      train_y->60000*1
# Test_x->10000*784       test_y->10000*1
filename_train_x = 'train-images-idx3-ubyte'
# train_x = Analytic_fun(filename_train_x)
Train_x = loadImageSet(filename_train_x)
filename_train_y = 'train-labels-idx1-ubyte'
#Train_y = Analytic_fun(filename_train_y)
Train_y = loadLabelSet(filename_train_y)
filename_test_x = 't10k-images-idx3-ubyte'
Test_x = loadImageSet(filename_test_x)
filename_test_y = 't10k-labels-idx1-ubyte'
Test_y = loadLabelSet(filename_test_y)
