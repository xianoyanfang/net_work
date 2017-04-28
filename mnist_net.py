# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 20:52:03 2017

@author: xiao
"""
import numpy as np
# 使用神经网络解决手写数字识别
y = OFK(Train_y,[0,1,2,3,4,5,6,7,8,9])
num = (784,100,10)
times = 10000
e = 1e-6
W,b = net_works_train(Train_x,y,num,times,e)
Y = net_works_test(Test_x,W,b)
pre_Y = OFK(Y,[0,1,2,3,4,5,6,7,8,9])
train_y = OFK(Test_y,[0,1,2,3,4,5,6,7,8,9])
accuary = 1 - np.sum(abs(pre_Y - train_y))/(2*len(Test_y))
print('accuary = ',accuary)