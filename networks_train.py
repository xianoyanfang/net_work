# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 08:53:06 2017

@author: xiao
"""
import numpy as np
# 神经网络训练集
def net_works_train(x,y,num,times,e):
    m,n = x.shape
    f = lambda x:1/(1+np.exp(-x)) # 这里使用sigmoid函数作为每一层的激活函数
    thx = 0.5 # thx 表示学习速率
    # 初始化学习参数
    W = []
    b = []
    res = []
    for i in range(0,len(num)-1):
        W.append(np.random.rand(num[i],num[i+1])) # 初始化 W
        b.append(np.random.rand(num[i+1],1)) # 初始化 b
        res.append(np.zeros((m,num[i+1])))
    
    
    
    for t in range(0,times):  
        a = []

        for i in range(0,len(num)):
            a.append(np.zeros((m,num[i])))
        a[0] = x
         # 前向传播
        for i in range(0,len(num)-1):
            a[i+1] = f(np.dot(a[i],W[i])+np.dot(np.ones((m,1)),b[i].T)) # 计算激活值            
            
        # 反向传播
        # 计算残差,感觉残差计算错误T_T，这里的残差值得关注
        for l in range(len(num)-2,-1,-1):# 1-0
            if l == len(num)-2:
                res[l] = (a[l+1] - y)*a[l+1]*(1-a[l+1])
            else:
                res[l] = np.dot(res[l+1],W[l+1].T)*a[l+1]*(1-a[l+1])
            # 更新网络参数
            W[l] = W[l] - thx*np.dot(a[l].T,res[l])
            # 这里 b 的更新存在问题
            b[l] = b[l] - thx*(np.sum(res[l],axis = 0)/m).reshape((num[l+1],1))
        if np.linalg.norm(a[len(num)-1]-y) < e:
            break
    W = W
    b = b
    return W,b
    
    
        
    
             
            
        
            