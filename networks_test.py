# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 20:57:52 2017

@author: xiao
"""
import numpy as np
def net_works_test(x,W,b):
    l = len(W)
    m,n = x.shape
    for i in range(0,l):
        x = f(np.dot(x,W[i]) + np.dot(np.ones((m,1)),b[i].T))
    index = np.argmax(x,axis = 1)
    return index
    