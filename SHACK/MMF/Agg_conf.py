# -*- codding: utf-8 -*-
'''
@Author : Yuren
@Dare   : 2022/3/30-1:38 下午

'''

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import os
from scipy.optimize import linear_sum_assignment as linear_assignment
import joblib
import time
import matplotlib.pyplot as plt
from agg_tools import *
def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    match = linear_assignment(w.max() - w)
    a , b = match
    suma = 0
    for i in range(len(a)):
        suma += w[a[i],b[i]]
    return suma*1.0/y_pred.size ,match
def conf(y_true,y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(),y_true.max())+1
    w = np.zeros((D,D),dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i],y_true[i]]+=1
    match = linear_assignment(w.max()-w)
    a , b = match
    confuse = np.zeros((D,D),dtype=np.int64)
    for i in range(256):
        confuse[b[i],:]=w[a[i],:]
    return confuse
now_label = []
for i in range(256):
    for j in range(100):
        now_label.append(i)

true_label = np.array(now_label)

Aggmodel = joblib.load('agg_model/64iter_98_Agg_0.97_.pkl')
pre_label = Aggmodel.labels_

accuracy,match = acc(true_label,pre_label)
print('model acc:{}'.format(accuracy))
confuse = conf(true_label,pre_label)
#print(confuse[0:25,0:25])
#for i in range(12):
#    print(confuse[i*20:(i+1)*20,i*20:(i+1)*20])

#print(confuse[240:255,240:255])



print(confuse[42:52,42:52])
